#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::{DType, IntDType, Module, Tensor};
use candle_nn::{embedding, ops::softmax, VarBuilder};
use candle_transformers::models::deberta_v2::{Config, DebertaV2Model, HiddenAct, DTYPE};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    premise: Option<String>,

    #[arg(long)]
    hypotesis: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation.
    #[arg(long, default_value = "false")]
    approximate_gelu: bool,
}

pub struct MDeberta {
    deberta_model: DebertaV2Model,
    pooler: candle_nn::linear::Linear,
    classifier: candle_nn::linear::Linear,
    tokenizer: Tokenizer,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<MDeberta> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7".to_string();
        let default_revision = "main".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config_json = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config_json)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_json)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let pooler_hidden_size: usize = config_json["pooler_hidden_size"]
            .as_i64()
            .unwrap()
            .as_usize();

        let deberta_model = DebertaV2Model::load(vb.pp("deberta"), &config)?;
        let pooler = candle_nn::linear(
            pooler_hidden_size,
            pooler_hidden_size,
            vb.pp("pooler.dense"),
        )?;
        let classifier = candle_nn::linear(pooler_hidden_size, 3, vb.pp("classifier"))?;

        let mdeberta = MDeberta {
            deberta_model,
            pooler,
            classifier,
            tokenizer,
        };
        Ok(mdeberta)
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let start = std::time::Instant::now();

    let mdeberta = args.build_model_and_tokenizer()?;
    let deberta_model = mdeberta.deberta_model;
    let mut tokenizer = mdeberta.tokenizer;
    let device = &deberta_model.device;

    let premise = if let Some(p) = args.premise {
        p
    } else {
        "The cat sleeps on the windowsill.".to_string()
    };
    let hypothesis = if let Some(h) = args.hypotesis {
        h
    } else {
        "All the cats are outside now.".to_string()
    };

    println!("PREMISE: {premise}");
    println!("HYPOTHESIS: {hypothesis}");

    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let tokens = tokenizer
        .encode((premise, hypothesis), true)
        .map_err(E::msg)?;

    let token_ids = tokens.get_ids().to_vec();
    let token_type_ids = tokens.get_type_ids().to_vec();

    //println!("{token_ids:?}");
    //println!("{token_type_ids:?}");

    let token_ids = Tensor::new(&token_ids[..], device)?.unsqueeze(0)?;
    let token_type_ids = Tensor::new(&token_type_ids[..], device)?.unsqueeze(0)?;
    println!("Loaded and encoded {:?}", start.elapsed());
    let start = std::time::Instant::now();

    let hidden_states = deberta_model.forward(&token_ids, &token_type_ids)?;
    println!("{hidden_states}");

    let context_token = hidden_states.get_on_dim(1, 0)?;

    // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    //let (_n_sentence, n_tokens, _hidden_size) = hidden_states.dims3()?;
    //let context_token = (hidden_states.sum(1)? / (n_tokens as f64))?;

    println!("{context_token}");
    let pooled_output = mdeberta.pooler.forward(&context_token)?;
    let pooled_output = pooled_output.gelu_erf()?;
    let output = mdeberta.classifier.forward(&pooled_output)?;

    println!("{output:?}");

    let prediction = softmax(&output, 1)?;
    let prediction: Vec<f32> = prediction.squeeze(0)?.to_vec1()?;
    let label_names = ["entailment", "neutral", "contradiction"];

    for (ix, label) in label_names.iter().enumerate() {
        let pred = prediction[ix];
        println!("{label:?} {pred:?}");
    }

    Ok(())
}
