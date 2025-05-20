use burn::{
            data::{dataloader::batcher::Batcher, 
                dataset::{transform::PartialDataset, Dataset, InMemDataset}}, 
            prelude::*};
use serde::{Deserialize, Serialize};

type PartialData = PartialDataset<InMemDataset<ClientItem>, ClientItem>;

const FITTED_LAMBDA_INCOME: f32 = 0.3026418664067109;
const FITTED_LAMBDA_WEALTH: f32 =  0.1336735055366279;
const SCALER_MEANS: [f32; 9] = [55.2534, 0.492, 2.5106, 0.41912250425, 7.664003606794818, 5.650795740523163, 0.3836, 0.5132, 1.7884450179129336];
const SCALER_SCALES: [f32; 9] = [11.970496582849016, 0.49993599590347565, 0.7617661320904205, 0.15136821202756753, 2.4818937424620633, 1.5813522545815777, 0.4862623160393987, 0.49982572962983807, 0.8569630982206199];

#[derive(Clone, Deserialize, Serialize, Debug, Copy)]
pub struct ClientItem {
    pub(crate) row_id: i32,
    pub(crate) age: i8,
    pub(crate) gender: i8,
    pub(crate) family_members: i8,
    pub(crate) financial_education: f32,
    pub(crate) risk_propensity: f32,
    pub(crate) income: f32,
    pub(crate) wealth: f32,
    pub(crate) income_investment: i8,
    pub(crate) accumulation_investment: i8,
    pub(crate) client_id: i8,
}

#[derive(Clone, Debug)]
pub struct ClientBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ClientBatch<B: Backend> {
    pub data: Tensor<B, 2>,
    pub target: Tensor<B, 1>,
}

impl<B: Backend> ClientBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, Vec<f32>, ClientBatch<B>> for ClientBatcher<B> {
    fn batch(&self, items: Vec<Vec<f32>>, device: &B::Device) -> ClientBatch<B> {
      
        let data = items.clone()
            .iter()
            .map(move |v| {
                let mut v = v.clone();
                v.pop();
                Tensor::<B, 2>::from_data(TensorData::new(v.clone(), [1, v.len()]), &self.device)})
            .collect();

        let target = items.clone()
            .iter()
            .map(|item| {
                let risk_propensity = item.clone().pop().unwrap();
                Tensor::<B, 1>::from_floats([risk_propensity], &self.device)})
            .collect();

        let data = Tensor::cat(data, 0);
        let target = Tensor::cat(target, 0);

        ClientBatch { data, target }
    }
}

pub struct ClientDataset {
    dataset: InMemDataset<ClientItem>,
}

impl Clone for ClientDataset {
    fn clone(&self) -> Self {
        let mut v = Vec::new();
        for i in 0..self.dataset.len() {
            v.push(self.dataset.get(i).unwrap());
        }
        Self { dataset: InMemDataset::<ClientItem>::new(v) }
    }
}

impl ClientDataset {
    pub fn new(file_path: &str) -> Result<Self, std::io::Error> {
        // Build dataset from csv
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b',');

        let dataset = InMemDataset::from_csv(file_path, rdr)?;

        let dataset = Self { dataset };
        Ok(dataset)
    }

    pub fn from_InMemD(dataset: InMemDataset<ClientItem>) -> Self {
        Self { dataset: dataset }
    }

    pub fn dataset(&self) -> InMemDataset<ClientItem> {
        let mut v = Vec::new();
        for i in 0..self.dataset.len() {
            v.push(self.dataset.get(i).unwrap());
        }
        InMemDataset::<ClientItem>::new(v)
    }
}


// Implement the `Dataset` trait which requires `get` and `len`
impl Dataset<ClientItem> for ClientDataset {
    fn get(&self, index: usize) -> Option<ClientItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub fn get_train_test(dataset: ClientDataset, split: &str) -> ClientDataset {
        // define chained dataset type here for brevity
    let len = dataset.dataset.len();

    match split {
        "train" => ClientDataset::from_InMemD(InMemDataset::from_dataset(&PartialData::new(dataset.dataset, 0, len * 8 / 10))),
        "test" => ClientDataset::from_InMemD(InMemDataset::from_dataset(&PartialData::new(dataset.dataset, len * 8 / 10, len))),
        _ => panic!("Error")
    }
}

pub fn preprocessing(dataset: ClientDataset) -> InMemDataset<Vec<f32>> {
    // Preprocess the client data
    // Convert the client data into a vector of f32

    let mut client_vec: Vec<Vec<f32>> = Vec::new();

    for client in dataset.iter() {
        
        let v = vec![
            client.age as f32,
            client.gender as f32,
            client.family_members as f32,
            client.financial_education,
            (client.income.powf(FITTED_LAMBDA_INCOME) - 1.0) / FITTED_LAMBDA_INCOME,
            (client.wealth.powf(FITTED_LAMBDA_WEALTH) - 1.0) / FITTED_LAMBDA_WEALTH,
            client.income_investment as f32,
            client.accumulation_investment as f32,
            client.financial_education * client.wealth.ln(),
        ];
        // Scale the data using means and scales
        let mut a: Vec<f32> = v
            .iter()
            .zip(SCALER_MEANS.iter().zip(SCALER_SCALES.iter()))
            .map(|(&value, (&mean, &scale))| (value - mean) / scale)
            .collect();

        a.push(client.risk_propensity);

        client_vec.push(a);
    }

    InMemDataset::new(client_vec)
}
