use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
};

// Número de palavras no vocabulário (bag of words)
pub const VOCAB_SIZE: usize = 64;
// Número de categorias: 0=normal, 1=urgent, 2=spam
pub const NUM_CLASSES: usize = 3;

#[derive(Module, Debug)]
pub struct MessageClassifier<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> MessageClassifier<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(VOCAB_SIZE, 128).init(device),
            fc2: LinearConfig::new(128, 64).init(device),
            fc3: LinearConfig::new(64, NUM_CLASSES).init(device),
            dropout: DropoutConfig::new(0.3).init(),
        }
    }

    /// Forward pass: recebe tensor [batch, VOCAB_SIZE], retorna logits [batch, NUM_CLASSES]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.dropout.forward(x);

        let x = self.fc2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.dropout.forward(x);

        self.fc3.forward(x) // logits crus — aplicar softmax na inferência
    }

    /// Inferência: retorna (categoria, score de confiança)
    pub fn predict(&self, x: Tensor<B, 2>) -> (usize, f32) {
        let logits = self.forward(x);
        let probs = burn::tensor::activation::softmax(logits, 1);

        // Extrai os valores como Vec<f32>
        let data = probs.into_data();
        let values: Vec<f32> = data.convert::<f32>().to_vec().unwrap();

        // Pega o índice de maior probabilidade
        let (class_idx, &score) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        (class_idx, score)
    }
}

/// Mapeia índice de classe para string
pub fn class_label(idx: usize) -> &'static str {
    match idx {
        0 => "normal",
        1 => "urgent",
        2 => "spam",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    #[test]
    fn test_forward_pass_shape() {
        let device = Default::default();
        let model = MessageClassifier::<Backend>::new(&device);

        // batch de 2 mensagens, VOCAB_SIZE features cada
        let input = Tensor::<Backend, 2>::zeros([2, VOCAB_SIZE], &device);
        let output = model.forward(input);

        // output deve ser [2, NUM_CLASSES]
        assert_eq!(output.shape().dims, [2, NUM_CLASSES]);
    }

    #[test]
    fn test_predict_returns_valid_class() {
        let device = Default::default();
        let model = MessageClassifier::<Backend>::new(&device);

        let input = Tensor::<Backend, 2>::zeros([1, VOCAB_SIZE], &device);
        let (class_idx, score) = model.predict(input);

        assert!(class_idx < NUM_CLASSES);
        assert!(score > 0.0 && score <= 1.0);
    }
}
