use csv::Reader;
use puffpastry::*;
use rand::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json;
use std::fs::File;

#[allow(unused)]
pub enum DataType {
    Mnist,
    MnistFakes,
}

pub fn train_and_serialize_model(data_type: DataType) -> File {
    let model = match data_type {
        DataType::Mnist => get_mnist_trained(),
        DataType::MnistFakes => get_mnist_trained_fakes(),
    };

    let model_json = serde_json::to_string(&Helper(model)).expect("Error writing model to string");
    std::fs::write("data/model.json", model_json).expect("Error writing file!");

    File::open("data/model.json").unwrap()
}

fn get_mnist_trained_fakes() -> Model<f64> {
    let mut model = Model {
        layers: vec![Layer::from_size(784, 11, Activation::Softmax)],
        loss: Loss::CategoricalCrossEntropy,
    };

    let (train, validate) = mnist_with_fakes(60000);
    model.fit(train, validate, 3, 0.002);

    model
}

fn get_mnist_trained() -> Model<f64> {
    let mut model = Model {
        layers: vec![Layer::from_size(784, 10, Activation::Softmax)],
        loss: Loss::CategoricalCrossEntropy,
    };

    let (train, validate) = load_mnist();
    model.fit(train, validate, 3, 0.002);

    model
}

pub fn create_fakes(count: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();
    let mut validate = vec![0.0; 11];
    validate[10] = 1.0;
    let fake_pixels = |_| (0..784).map(|_| rng.gen_range(0.0..1.0)).collect();

    let fake_train = (0..count).map(fake_pixels).collect();
    let fake_validate = (0..count).map(|_| validate.clone()).collect();
    (fake_train, fake_validate)
}

pub fn mnist_with_fakes(fake_count: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let (mut fake_train, mut fake_validate) = create_fakes(fake_count);
    let mut train: Vec<Vec<f64>> = vec![];
    let mut validate: Vec<Vec<f64>> = vec![];
    let mut mnist_reader = Reader::from_path("data/mnist_train.csv").unwrap();
    let labels = 11;

    for (_, record) in mnist_reader.records().enumerate() {
        if let Ok(record) = record {
            let label: usize = record
                .into_iter()
                .take(1)
                .map(|x| x.parse().unwrap())
                .next()
                .unwrap();

            let mut val_vec = vec![0.0; labels];
            val_vec[label] = 1.0;
            let train_vec = record
                .into_iter()
                .skip(1)
                .map(|x| x.parse::<f64>().unwrap() / 255.0)
                .collect();

            validate.push(val_vec);
            train.push(train_vec);
        }
    }

    train.append(&mut fake_train);
    validate.append(&mut fake_validate);

    (train, validate)
}

pub fn load_mnist() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut train: Vec<Vec<f64>> = vec![];
    let mut validate: Vec<Vec<f64>> = vec![];
    let mut mnist_reader = Reader::from_path("data/mnist_train.csv").unwrap();
    let labels = 10;

    for (_, record) in mnist_reader.records().enumerate() {
        if let Ok(x) = record {
            let label: usize = x
                .into_iter()
                .take(1)
                .map(|x| x.parse().unwrap())
                .next()
                .unwrap();

            let mut val_vec = vec![0.0; labels];
            val_vec[label] = 1.0;
            validate.push(val_vec);

            train.push(
                x.into_iter()
                    .skip(1)
                    .map(|x| x.parse::<f64>().unwrap() / 255.0)
                    .collect(),
            );
        }
    }

    (train, validate)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Helper(#[serde(with = "ModelDef")] pub Model<f64>);

#[derive(Serialize, Deserialize)]
#[serde(remote = "Model<f64>")]
struct ModelDef {
    #[serde(deserialize_with = "deserialize_vec_layer")]
    #[serde(serialize_with = "serialize_vec_layer")]
    layers: Vec<Layer<f64>>,
    #[serde(with = "LossDef")]
    loss: Loss,
}

fn deserialize_vec_layer<'de, D>(deserializer: D) -> Result<Vec<Layer<f64>>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct Wrapper(#[serde(with = "LayerDef")] Layer<f64>);

    let v = Vec::deserialize(deserializer)?;
    Ok(v.into_iter().map(|Wrapper(a)| a).collect())
}

fn serialize_vec_layer<S>(items: &Vec<Layer<f64>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    #[derive(Serialize)]
    struct Wrapper(#[serde(with = "LayerDef")] Layer<f64>);

    Ok(items
        .into_iter()
        .map(|x| Wrapper(x.clone()))
        .collect::<Vec<Wrapper>>()
        .serialize(serializer)?)
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Layer<f64>")]
struct LayerDef {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    #[serde(with = "ActivationDef")]
    activation: Activation,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Activation")]
enum ActivationDef {
    None,
    Sigmoid,
    Relu,
    Softmax,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Loss")]
enum LossDef {
    MeanSquaredError,
    CategoricalCrossEntropy,
}
