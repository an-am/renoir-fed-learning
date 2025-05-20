use burn::{
            data::{dataloader::DataLoaderBuilder, dataset::InMemDataset}, 
            module::AutodiffModule, 
            optim::{AdamConfig, GradientsParams, Optimizer}, 
            prelude::*, 
            record::{BinBytesRecorder, FullPrecisionSettings, Recorder}, 
            tensor::backend::AutodiffBackend, 
};

use nn::loss::{MseLoss, Reduction};

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

pub fn local_train<B: AutodiffBackend>(device: B::Device, config: ClientTrainingConfig) -> Result<Vec<u8>, burn::record::RecorderError> {
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
    for epoch in 1..config.num_epochs + 1 {
        // Training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.data);
            let loss = MseLoss::new().forward(output, batch.target.unsqueeze(), Reduction::Mean);
            
            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }

        // Get the model without autodiff.
        let model_valid = model.valid();

        // Validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.data);
            let loss = MseLoss::new().forward(output, batch.target.unsqueeze(), Reduction::Mean);
            
            println!(
                "[Valid - Epoch {} - Iteration {}] Loss {}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            ); 
            
        }
    }

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    recorder.record(model.into_record(), ())
}

