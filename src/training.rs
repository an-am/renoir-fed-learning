use burn::{
            data::{dataloader::DataLoaderBuilder, dataset::InMemDataset}, module::AutodiffModule, nn::loss, optim::{AdamConfig, GradientsParams, Optimizer}, prelude::*, record::{BinBytesRecorder, FullPrecisionSettings, Recorder, RecorderError}, tensor::{backend::AutodiffBackend, cast::ToElement, ops::FloatElem} 
};

use nn::loss::{MseLoss, Reduction};
use std::fs::File;
use csv::WriterBuilder;

use crate::{
                dataset::{ClientBatcher, ClientDataset}, 
                model::{Model, ModelConfig, ModelRecord}
            };

pub struct ClientTrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub lr: f64,
    pub train_set: InMemDataset<Vec<f32>>,
    pub test_set: InMemDataset<Vec<f32>>,
    pub model_record: Vec<u8>
}

impl ClientTrainingConfig {
    pub fn new(train: InMemDataset<Vec<f32>>, test: InMemDataset<Vec<f32>>, model_record: Vec<u8>) -> Self {
        ClientTrainingConfig { 
            num_epochs: 10, 
            batch_size: 64, 
            num_workers: 1, 
            seed: 42, 
            lr: 1e-4, 
            train_set: train, 
            test_set: test, 
            model_record: model_record
        }
    }
}

pub fn local_train<B: AutodiffBackend>(device: B::Device, config: ClientTrainingConfig, i: i32, iteration: usize) -> Result<Vec<u8>, RecorderError> {
    B::seed(config.seed);

    // Create the model and optimizer.
    let record: ModelRecord<B> = BinBytesRecorder::<FullPrecisionSettings>::new()
        .load(config.model_record.clone(), &device).expect(&format!("TRAINING: File {:?} not found", config.model_record));
    let mut model: Model<B> = ModelConfig::new().init(&device).load_record(record);
    let mut optim = AdamConfig::new().init();

// ----------------------------------------------------

     // Create the batcher.
    let batcher_train = ClientBatcher::<B>::new(device.clone());
    let batcher_valid = ClientBatcher::<B::InnerBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(config.train_set);


    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(config.test_set); 
// ----------------------------------------------------

    // Iterate over training and validation loop for X epochs.
    let mut a: Vec<(i32, usize, usize, f32)> = Vec::new();

    for epoch in 1..config.num_epochs + 1 {
        // Training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.data);
            let loss = MseLoss::new().forward(output, batch.target.unsqueeze(), Reduction::Mean);
            
           /*  println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            ); */

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }

        // Get the model without autodiff.
        let model_valid = model.valid();

        let mut sum = 0.0;
        let mut n   = 0usize;

        // Validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.data);
            let loss: Tensor<<B as AutodiffBackend>::InnerBackend, 1> = MseLoss::new().forward(output, batch.target.unsqueeze(), Reduction::Mean);
            
            sum += loss.clone().into_scalar().to_f32();
            n += 1;

            /* println!(
                "[Valid - Epoch {} - Iteration {}] Loss {}",
                epoch,
                iteration,
                loss.clone().into_scalar()
            );  */
        }

        let epoch_mse = sum / n as f32;
        a.push((i, iteration, epoch, epoch_mse));
    }
    
   /*  // ---------------- Save per‑epoch losses to CSV ----------------
    let file_name = format!("local_loss_{}_{}.csv", i, iteration);
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(&file_name)
        .expect("Cannot create CSV file");

    // header
    wtr.write_record(&["i", "iteration", "epoch", "mse"])
        .expect("Cannot write header");

    // rows
    for (ci, it, ep, mse) in &a {
        wtr.serialize((ci, it, ep, mse))
           .expect("Cannot write record");
    }
    wtr.flush().expect("Cannot flush writer");
    // --------------------------------------------------------------
 */
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    recorder.record(model.into_record(), ())
}
