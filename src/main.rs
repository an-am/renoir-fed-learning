use std::{fs::File, io::Write, sync::Arc};
use model::{Model, ModelRecord};
use renoir::prelude::*;
use dataset::{get_train_test, preprocessing, ClientDataset, ClientItem};
use burn::{
            backend::Autodiff, 
            data::dataset::{transform::PartialDataset, InMemDataset}, 
            module::{ConstantRecord, Module, Param}, 
            nn::LinearRecord, 
            record::{BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, Recorder}, 
            tensor::{Tensor, TensorData}
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

/* 
pub fn avg_record(records: Vec<ModelRecord<Autodiff<NdArray>>>) -> ModelRecord<Autodiff<NdArray>> {
    let r = &records[0];
    let mut weight_layer1: Tensor<Autodiff<NdArray>, 2> = Tensor::zeros_like(&r.layer1.weight);
    let mut weight_layer2: Tensor<Autodiff<NdArray>, 2> = Tensor::zeros_like(&r.layer2.weight);

    let mut bias_layer1: Tensor<Autodiff<NdArray>, 1> = Tensor::zeros_like(&r.layer1.bias.clone().unwrap().val());
    let mut bias_layer2: Tensor<Autodiff<NdArray>, 1> = Tensor::zeros_like(&r.layer2.bias.clone().unwrap().val());
    
    let n = records.len() as i32;
    
    for record in records {
        // Weight
        let x = record.layer1.weight.val();
        weight_layer1 = weight_layer1.add(x);

        let x = record.layer2.weight.val();
        weight_layer2 = weight_layer2.add(x);

        // Bias
        let x = record.layer1.bias.unwrap().val();
        bias_layer1 = bias_layer1.add(x);

        let x = record.layer2.bias.unwrap().val();
        bias_layer2 = bias_layer2.add(x);
    }

    let global_weight_layer1: Param<Tensor<Autodiff<NdArray>, 2>> = Param::from_tensor(weight_layer1.detach().div_scalar(n));
    let global_weight_layer2: Param<Tensor<Autodiff<NdArray>, 2>> = Param::from_tensor(weight_layer2.detach().div_scalar(n));

    let global_bias_layer1: Option<Param<Tensor<Autodiff<NdArray>, 1>>> = Some(Param::from_tensor(bias_layer1.detach().div_scalar(n)));
    let global_bias_layer2: Option<Param<Tensor<Autodiff<NdArray>, 1>>> = Some(Param::from_tensor(bias_layer2.detach().div_scalar(n)));

    let layer_1 = LinearRecord {
        weight: global_weight_layer1,
        bias: global_bias_layer1,
    };

    let layer_2 = LinearRecord {
        weight: global_weight_layer2,
        bias: global_bias_layer2,
    };

    ModelRecord::<Autodiff<NdArray>> {
        layer1: layer_1,
        layer2: layer_2,
        activation: ConstantRecord,
    }
}
*/

fn main() {
    let (conf, _args) = RuntimeConfig::from_args();
    conf.spawn_remote_workers();
    let conf = Arc::new(conf);

    // Create a device for tensor computation
    let device = NdArrayDevice::default();

    // Get train and test data
    let file_path: &str= "/Users/antonelloamore/VS code/renoir-fed-training/Needs.csv";

    let dataset = ClientDataset::new(file_path).unwrap();

    let train_dataset = get_train_test(dataset.clone(), "train");
    let test_dataset = get_train_test(dataset, "test");

    let train_partition = PartialDataset::split(train_dataset, 10);
    let test_partition = PartialDataset::split(test_dataset, 10);

    let mut vec1: Vec<ClientDataset> = Vec::new();
    let mut vec2: Vec<ClientDataset> = Vec::new();
    for i in 0..10 as usize { 
        vec1.push(ClientDataset::from_InMemD(InMemDataset::from_dataset(&train_partition[i])));
        vec2.push(ClientDataset::from_InMemD(InMemDataset::from_dataset(&test_partition[i])));
    }
    
    // Create the model
    let mut model: Model<MyAutoDiff> = model::ModelConfig::new().init(&device);

    let local_model_dir = "/Users/antonelloamore/VS code/renoir-fed-training/local-models/";

    for i in 0..10 {
        let local_model_path = format!("{}{}", local_model_dir, i);

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(local_model_path, &recorder);
    }

    // Train and valid sets
    let vec1: Arc<Vec<ClientDataset>> = Arc::new(vec1);
    let vec2: Arc<Vec<ClientDataset>> = Arc::new(vec2);

    for i in 0..N_ITERATIONS as usize {
        let ctx = StreamContext::new(conf.clone());

        let s = ctx
            .stream_par_iter(0..N_MODELS)
            .rich_map({
                let vec1 = Arc::clone(&vec1); 
                let vec2 = Arc::clone(&vec2);
                move |i| {
                    let local_model_path = format!("{}{}", local_model_dir, i);
                    let config = ClientTrainingConfig::new(
                        vec1[i as usize].clone(),
                        vec2[i as usize].clone(),
                        format!("{}", local_model_path)
                    );

                    local_train::<MyAutoDiff>(device.clone(), config).unwrap()
                } 
            })
            .fold(None, |acc: &mut Option<ModelRecord<Autodiff<NdArray>>>, record: Vec<u8>| {
                    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
                    let r: ModelRecord<Autodiff<NdArray>> = recorder.load::<ModelRecord<MyAutoDiff>>(record, &(NdArrayDevice::default())).unwrap();
                    
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
            .map(|a| {
                let ModelRecord { layer1, layer2, activation } = a.unwrap();
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

        let s = s.get().unwrap().pop().unwrap();    
        
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
        let global_record = recorder.load::<ModelRecord<MyAutoDiff>>(s, &device).unwrap();
        model = model.load_record(global_record);

        for i in 0..N_MODELS {
            let local_model_path = format!("{}{}", local_model_dir, i);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            
            model.clone().save_file(local_model_path, &recorder).expect(&format!("File {:?} not saved", format!("local_model{}", i)));
        }
        println!("Iteration {:?} completed!", i);
    }

   // TEST: inference
    let source = CsvSource::<ClientItem>::new("/Users/antonelloamore/VS code/renoir-prediction/Needs.csv");
    let env = StreamContext::new(conf);
    let s = env
        .stream(source)
        .map(|v| preprocessing(v))
        .map(move |v| Tensor::<MyAutoDiff, 2>::from_data(TensorData::new(v.clone(), [1, v.len()]), &device))
        .map(move |v| model.forward(v))
        .map(|v| v.to_data().to_vec::<f32>().unwrap())
        .collect_vec(); // for_each println!("Model output: {:?}", v.first().unwrap())); 

    env.execute_blocking();

    let mut file = File::create("/Users/antonelloamore/VS code/renoir-fed-training/local-models/file.txt").unwrap();
    let a = s.get().unwrap();
    
    for row in a {
        let line = row.iter()
            .map(|item| item.to_string())
            .collect::<Vec<String>>()
            .join(",");
        writeln!(file, "{}", line);
    }
}
