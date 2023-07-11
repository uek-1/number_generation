use clap::{arg, command, value_parser, ArgAction, Command};
use image::ImageBuffer;
use puffpastry::Model;
use rand::prelude::*;
use serde_json;
use std::env;
use std::fs::File;
use std::io::BufReader;

mod model_serialize;
use model_serialize::{train_and_serialize_model, Helper};

fn main() {
    let matches = command!()
        .arg(
            arg!(-s --retrain "Retrains model")
                .required(false)
                .action(ArgAction::SetTrue),
        )
        .arg(
            arg!(-n --num <NUM>)
                .required(true)
                .value_parser(|x: &str| x.parse::<usize>()),
        )
        .get_matches();

    let num: usize = *matches.get_one::<usize>("num").unwrap();

    if matches.get_flag("retrain") {
        train_and_serialize_model(model_serialize::DataType::MnistFakes);
    }

    let file = match File::open("data/model.json") {
        Ok(f) => f,
        Err(_) => train_and_serialize_model(model_serialize::DataType::MnistFakes),
    };

    let buffer = BufReader::new(file);
    let trained_model_obj: Helper = match serde_json::from_reader(buffer) {
        Ok(model) => model,
        Err(_) => panic!("model.json is corrupt!, delete file and rerun"),
    };

    let mut model = trained_model_obj.0;

    let mut rng = rand::thread_rng();
    let mut state: Vec<f64> = (0..784).map(|_| rng.gen_range(0.0..1.0)).collect();
    let mut target = vec![0.0; 11];
    target[num] = 1.0;
    let state_rate = 0.04;
    let initial_loss = get_loss_value(state.clone(), target.clone(), &model);
    let initial_res = model.evaluate(&state);
    let initial_max_res = argmax(&initial_res);
    println!("Intially: Loss {initial_loss} Predicted Class : {initial_max_res}\n\n Predications {initial_res:?}");

    let iter_path = |x: usize| format!("data/iterations/{x}.png");

    for i in 0..201 {
        let (state_loss, max_res, res) = train_state(
            &mut state,
            &target,
            &mut model,
            state_rate,
            0.005 * i as f64,
        );
        println!("\nIteration {i} : Loss : {state_loss} Predicted Class : {max_res}\n\n Model Predictions {res:?}\n");

        // pretty_print(&state);
        if i % 5 == 0 {
            write_to_image(&state, iter_path(i))
        }
    }

    create_gif()
}

fn finite_diff(state: &Vec<f64>, target: &Vec<f64>, model: &Model<f64>) -> Vec<f64> {
    state
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let mut inc_state = state.clone();
            inc_state[index] += 0.01;
            let current = get_loss_value(state.clone(), target.clone(), model);
            let inc = get_loss_value(inc_state.clone(), target.clone(), model);
            (inc - current) / 0.01
        })
        .collect()
}

fn get_loss_value(input: Vec<f64>, target: Vec<f64>, model: &Model<f64>) -> f64 {
    let res = model.evaluate(&input);
    model.loss.calculate_loss(res, target)
}

#[allow(unused)]
fn pretty_print(v: &Vec<f64>) {
    for i in 0..28 {
        for j in 0..28 {
            match v[i * 28 + j] > 0.25 {
                true => print!("1 "),
                false => print!(" "),
            };
        }
        println!()
    }
}

fn write_to_image(state: &Vec<f64>, path: String) {
    let mut img = ImageBuffer::new(28, 28);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let generated = (255.0 * state[(x + 28 * y) as usize]) as u8;
        *pixel = image::Rgb([generated, generated, generated]);
    }

    img.save(path).expect("Couldn't write image to path!");
}

fn argmax(preds: &Vec<f64>) -> usize {
    let (idx, _) = preds
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    idx
}

fn train_state(
    state: &mut Vec<f64>,
    target: &Vec<f64>,
    model: &mut Model<f64>,
    state_rate: f64,
    model_rate: f64,
) -> (f64, usize, Vec<f64>) {
    let diffs = finite_diff(state, target, model);
    *state = state
        .iter()
        .enumerate()
        .map(|(index, pixel)| pixel - (diffs[index] * state_rate))
        .collect();

    let temp_train = vec![state.clone()];
    let temp_truth = vec![(0..11).map(|x| if x == 10 { 1.0 } else { 0.0 }).collect()];

    model.fit(temp_train.clone(), temp_truth.clone(), 1, model_rate);

    let res = model.evaluate(&state);
    let state_loss = get_loss_value(state.clone(), target.clone(), &model);
    let max_res = argmax(&res);

    (state_loss, max_res, res)
}

fn create_gif() {
    let mut buf = std::fs::File::create("data/iterations/all.gif").expect("couldn't create gif");
    let mut enc = image::codecs::gif::GifEncoder::new_with_speed(buf, 10);
    let mut frames = vec![];
    let path = |x: usize| format!("data/iterations/{x}.png");

    for i in 0..100 {
        if let Ok(img) = image::open(path(5 * i)) {
            let frame = image::Frame::new(img.into_rgba8());
            frames.push(frame)
        }
    }
    enc.set_repeat(image::codecs::gif::Repeat::Infinite);
    enc.encode_frames(frames);
}
