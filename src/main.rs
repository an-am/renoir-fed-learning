use std::sync::Arc;

use csv::WriterBuilder;
use model::{Model, ModelRecord};
use renoir::{prelude::*};
use dataset::{get_train_test, preprocessing, ClientDataset};
use burn::{
            backend::Autodiff, 
            data::dataset::{transform::PartialDataset, Dataset, InMemDataset}, 
            module::{ConstantRecord, Module, Param}, 
            nn::LinearRecord, 
            record::{BinBytesRecorder, FullPrecisionSettings, Recorder}
        };

use training::{local_train, ClientTrainingConfig};
use burn_ndarray::{NdArray, NdArrayDevice};

mod model;
mod dataset;
mod training;

type MyBackend = NdArray<f32>;
type MyAutoDiff = Autodiff<MyBackend>;

const N_MODELS: i32 = 10;
const N_ITERATIONS: i32 = 10;

fn main() {
    let (conf, _args) = RuntimeConfig::from_args();
    conf.spawn_remote_workers();

    // Create a device for tensor computation
    let device = NdArrayDevice::default();

    // Get train and test data
    let file_path: &str= "/Users/antonelloamore/VScode/renoir-fed-training/Needs.csv";

    let dataset = ClientDataset::new(file_path).unwrap();

    let train_dataset = get_train_test(dataset.clone(), "train");
    let test_dataset = get_train_test(dataset, "test");

    let train_partition = PartialDataset::split(train_dataset, N_MODELS);
    let test_partition = PartialDataset::split(test_dataset, N_MODELS);

    let mut train: Vec<ClientDataset> = Vec::new();
    let mut test: Vec<ClientDataset> = Vec::new();
    for i in 0..10 as usize { 
        train.push(ClientDataset::from_InMemD(InMemDataset::from_dataset(&train_partition[i])));
        test.push(ClientDataset::from_InMemD(InMemDataset::from_dataset(&test_partition[i])));
    }
    
    // Create the model
    let mut global_model: Model<MyAutoDiff> = model::ModelConfig::new().init(&device);

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let mut global_record: Vec<u8> = recorder.record(global_model.clone().into_record(), ()).unwrap();

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
        })
        .collect_vec_all();
    
    ctx.execute_blocking();

    println!("ok data prep");
    let data= s.get().unwrap();
    
    for iteration in 0..N_ITERATIONS as usize {

        let ctx = StreamContext::new(conf.clone());

        let s = ctx
            .stream_par_iter( 0..N_MODELS)
            .rich_map({
                let data = Arc::new(data.clone());
                let local_record = global_record.clone();
                
                move|i| {
                    let data = data.get(i as usize).unwrap();
                    let train = InMemDataset::new(data.clone().0);
                    let test = InMemDataset::new(data.clone().1);
                    
                    let config = ClientTrainingConfig::new(train, test, local_record.clone());

                    // println!("Training model {:?}", i);
                    
                    local_train::<MyAutoDiff>(device.clone(), config, i, iteration).unwrap()
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
            .map(|record: Option<ModelRecord<Autodiff<NdArray>>>| {
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
            .collect_vec_all(); 
        
        let start_time = std::time::Instant::now();
        ctx.execute_blocking();
        let elapsed = start_time.elapsed();
        println!("Time: {:?}", elapsed);
       
        /* // --------------------------------------------------------------
        let file_name = format!("local_train_time_{}.csv", iteration);
        let mut wtr = WriterBuilder::new()
            .has_headers(true)
            .from_path(&file_name)
            .expect("Cannot create CSV file");

        // header
        wtr.write_record(&["iteration", "time"])
            .expect("Cannot write header");

        // rows
        wtr.serialize((iteration, elapsed.as_secs_f32()))
        .expect("Cannot write record");
        
        wtr.flush().expect("Cannot flush writer");
        // -------------------------------------------------------------- */

        global_record = s.get().unwrap().pop().unwrap();
        
        println!("Iteration {:?} completed!", iteration);
    }
    //println!("Training completed!");
    return;
}
