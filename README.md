# Bonfire
Run open-source generative AI models in a lightweight, reliable, and customizable rust API. \
This is a Rust project powered by Huggingface's Candle and Tokio's Axum. It focuses on text generation and image generation models.

## Usage
1. Install [Rust](https://www.rust-lang.org/tools/install)
2. Clone this repo
3. Set the 'API_ADDRESS' and optionally the 'API_PORT' environment variables to the desired address and port
4. Run `cargo run --release`

## API
### Text Generation
disclaimer: This program is designed to run on limited hardware, prompts often take upwards of a minute to finish generating. This API chooses the prompt/polling stragety for standalone prompts to avoid http requests timing out due to a long async process on the server. 
#### ```POST /prompt_polled```
Generates text from a prompt, stored on the server for a limited amount of time, to be polled. Returns the id of your content. \
**Parameters:**
- model: string
- prompt: string
- sample_len: integer
#### ```GET /poll_text/{id}```
Polls the server for the generated text. Returns the generated text or an error if the id is invalid or the content has expired. \
**Parameters:**
- id: string
#### ```POST /new_streaming```
Initializes a model and sets up message history for a new streaming session. Returns the id of your content. \
**Parameters:**
- model: string
#### ```POST /prompt_streaming/{id} (Unstable Work in Progress)```
Generates text from a prompt and streams it token by token to the consumer, stores message history on the server for a limited amount of time \
**Parameters:**
- id: string
- prompt: string
- sample_len: integer

## Supported Models
### Text Generation
- Mistral7b, 
- Mistral7b Instruct,
- Mistral7b Instruct V02,
- Mixtral (needs beefy gpu),
- Mixtral Instruct (needs beefy gpu),

- Mistral7b Quantized,
- Mistral7b Instruct Quantized,
- Mistral7b Instruct V02 Quantized,
- Mixtral Quantized,
- Mixtral Instruct Quantized,
- Zephyr Alpha Quantized (fine tuned mixtral),
- Zephyr Beta Quantized (fine tuned mixtral),
- Dolphin Mixtral Quantized (fine tuned mixtral),

- other llms coming soon...

### Image Generation
- coming soon...

## Extending
### Text Generation
1. Create a script that implements the following trait, where tokens are streamed into the mpsc Sender:
```rust
pub trait TextGeneratorInner: Send + Sync {
    fn run(&mut self, prompt: &str, sample_len: u32, sender: Sender<String>) -> anyhow::Result<()>;
}
```
2. Implement a factory function that loads the model into memory, this is part of the trait because by design this function should take custom arguments of your choice, or none at all:
```rust
impl YourModel {
    pub fn new(arguments_of_your_choice: Args, or_none_at_all: Option<Args>) -> anyhow::Result<Self> {
        ...
    }
}
```
3. Add the model to the `TextGenerationModel` enum in [src/text_generation/utils.rs](src/text_generation/utils.rs), this is what http requests will identify your model as:
```rust
pub enum TextGenerationModel {
    YourModelName,
    ...
}
```
4. Add the model name and factory function to the `match` statement in [src/text_generation/utils.rs](src/text_generation/utils.rs):
```rust
impl TextGenerator {
    pub fn new(model: TextGenerationModel, args: &TextGenerationArgs) -> anyhow::Result<Self> {
        match model {
            TextGenerationModel::YourModelName => wrap(YourModel::new(args)?),
            ...
        }
    }
}
```
5. Done! You can now use your model in the API.

### Image Generation
coming soon ...