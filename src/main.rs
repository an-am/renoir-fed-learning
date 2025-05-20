use std::{io::Error, sync::Arc};

use model::{Model, ModelRecord};
use renoir::prelude::*;
use dataset::{get_train_test, preprocessing, ClientBatcher, ClientDataset};
use burn::{
            backend::Autodiff, 
            data::{self, dataloader::{DataLoader, DataLoaderBuilder}, dataset::{transform::PartialDataset, Dataset, InMemDataset}}, 
            module::{ConstantRecord, Module, Param}, 
            nn::LinearRecord, 
            record::{BinBytesRecorder, FullPrecisionSettings, Recorder}, train
        };

use serde::{ser::SerializeTuple, Serialize};
use training::{local_train, ClientTrainingConfig};
use burn_ndarray::{NdArray, NdArrayDevice};

mod model;
mod dataset;
mod training;

type MyBackend = NdArray<f32>;
type MyAutoDiff = Autodiff<MyBackend>;

const N_MODELS: i32 = 10;
const N_ITERATIONS: i32 = 10;
const BATCH_SIZE: usize = 64;
const N_WORKERS: usize = 1;
const SEED: u64 = 42;

fn main() {
    let (conf, _args) = RuntimeConfig::from_args();
    conf.spawn_remote_workers();
    let conf = Arc::new(conf);

    // Create a device for tensor computation
    let device = NdArrayDevice::default();

    // Get train and test data
    let file_path: &str= "/Users/antonelloamore/VScode/renoir-fed-training/Needs.csv";

    let dataset = ClientDataset::new(file_path).unwrap();

    let train_dataset = get_train_test(dataset.clone(), "train");
    let test_dataset = get_train_test(dataset, "test");

    let train_partition = PartialDataset::split(train_dataset, 10);
    let test_partition = PartialDataset::split(test_dataset, 10);

    let mut train: Vec<ClientDataset> = Vec::new();
    let mut test: Vec<ClientDataset> = Vec::new();
    for i in 0..10 as usize { 
        train.push(ClientDataset::from_InMemD(InMemDataset::from_dataset(&train_partition[i])));
        test.push(ClientDataset::from_InMemD(InMemDataset::from_dataset(&test_partition[i])));
    }
    
    // Create the model
    let mut model: Model<MyAutoDiff> = model::ModelConfig::new().init(&device);

    let mut record_vec = Vec::new();
    for _ in 0..10 {
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        let bytes = recorder.record(model.clone().into_record(), ()).unwrap();
        record_vec.push(bytes);
    }

    // Train and valid sets
    let train: Arc<Vec<ClientDataset>> = Arc::new(train);
    let test: Arc<Vec<ClientDataset>> = Arc::new(test);

    // Data preparation
    let ctx = StreamContext::new(conf.clone());


    let s = ctx.stream_par_iter(0..N_MODELS)
        .map(move |i| {
            let binding = train.clone();
            let train = binding.get(i as usize).unwrap();
            let binding = test.clone();
            let test = binding.get(i as usize).unwrap();

            let x: InMemDataset<Vec<f32>>= preprocessing(train.clone());
            let y: InMemDataset<Vec<f32>> = preprocessing(test.clone());

            let mut x_vec: Vec<Vec<f32>> = Vec::new();
            let mut y_vec: Vec<Vec<f32>> = Vec::new();
            
            for i in x.iter() {
                x_vec.push(i.clone());
            }
            for i in y.iter() {
                y_vec.push(i.clone());
            }
            (x_vec, y_vec)
        }).collect_vec_all();

    ctx.execute_blocking();

    let data= s.get().unwrap();

    for iteration in 0..N_ITERATIONS as usize {

        let ctx = StreamContext::new(conf.clone());

        let local_record_vec: Vec<Vec<u8>> = record_vec.clone();

        let s = ctx
            .stream_par_iter( 0..N_MODELS)
            .rich_map({
                let data = Arc::new(data.clone());
                let local_record_vec = local_record_vec.clone();
                move|i| {
                    let binding = data.clone();
                    let data = binding.get(i as usize).unwrap();
                    let train = InMemDataset::new(data.clone().0);
                    let test = InMemDataset::new(data.clone().1);
                    let config = ClientTrainingConfig::new(
                        train,
                        test,
                        local_record_vec[i as usize].clone()
                    );
                    println!("Training model {:?}", i);
                    local_train::<MyAutoDiff>(device.clone(), config).unwrap()
                } 
            })
            .fold(None, |acc: &mut Option<ModelRecord<Autodiff<NdArray>>>, record: Vec<u8>| {
                    let recorder: BinBytesRecorder<FullPrecisionSettings> = BinBytesRecorder::<FullPrecisionSettings>::new();
                    let r: ModelRecord<Autodiff<NdArray>> = recorder.load(record, &(NdArrayDevice::default())).unwrap();
                    
                    if acc.is_none() {
                        acc.replace(r);
                    } 
                    else {
                        let acc_r: ModelRecord<Autodiff<NdArray>> = acc.clone().unwrap();

                        // Weight
                        let mut weight_layer1 = acc_r.layer1.weight.val();
                        weight_layer1 = weight_layer1.add(r.layer1.weight.val());

                        let mut weight_layer2 = acc_r.layer2.weight.val();
                        weight_layer2 = weight_layer2.add(r.layer2.weight.val());

                        // Bias
                        let mut bias_layer1 = acc_r.layer1.bias.unwrap().val();
                        bias_layer1 = bias_layer1.add(r.layer1.bias.unwrap().val());

                        let mut bias_layer2 = acc_r.layer2.bias.unwrap().val();
                        bias_layer2 = bias_layer2.add(r.layer2.bias.unwrap().val());
                        
                        let record = ModelRecord {
                            layer1: LinearRecord {
                            weight: Param::from_tensor(weight_layer1.detach()),
                            bias: Some(Param::from_tensor(bias_layer1.detach())),
                            },
                            layer2: LinearRecord {
                                weight: Param::from_tensor(weight_layer2.detach()),
                                bias: Some(Param::from_tensor(bias_layer2.detach())),
                            },
                            activation: ConstantRecord,
                            };
                        acc.replace(record); 
                    }
            })
            .map(|record| {
                let ModelRecord { layer1, layer2, activation: _ } = record.unwrap();
                let LinearRecord { weight: weight_l1, bias: bias_l1 } = layer1;
                let LinearRecord { weight: weight_l2, bias: bias_l2 } = layer2;

                let global_weight_layer1= weight_l1.val().detach().div_scalar(N_MODELS);
                let global_weight_layer2= weight_l2.val().detach().div_scalar(N_MODELS);
            
                let global_bias_layer1 = bias_l1.unwrap().val().detach().div_scalar(N_MODELS);
                let global_bias_layer2 = bias_l2.unwrap().val().detach().div_scalar(N_MODELS);
            
                let record = ModelRecord {
                    layer1: LinearRecord {
                        weight: Param::from_tensor(global_weight_layer1.detach()),
                        bias: Some(Param::from_tensor(global_bias_layer1.detach())),
                    },
                    layer2: LinearRecord {
                        weight: Param::from_tensor(global_weight_layer2.detach()),
                        bias: Some(Param::from_tensor(global_bias_layer2.detach())),
                    },
                    activation: ConstantRecord,
                };

                let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
                
                recorder.record(record, ()).unwrap()
            })
            .collect_vec();
            
        ctx.execute_blocking();
        
        let s = s.get();

        if !s.is_none() {
            println!("Loading model {:?}", iteration);
            let s = s.unwrap().pop().unwrap();
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
            let global_record = recorder.load::<ModelRecord<MyAutoDiff>>(s, &device).unwrap();
            model = model.load_record(global_record);
            record_vec.clear();
            
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
            let bytes = recorder.record(model.clone().into_record(), ()).unwrap();
            
            for _ in 0..N_MODELS {
                record_vec.push(bytes.clone());
            }
        }

        println!("Iteration {:?} completed!", iteration);
    }
    println!("Training completed!");
    return;
}