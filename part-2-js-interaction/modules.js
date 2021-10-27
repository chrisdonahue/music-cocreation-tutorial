window.my = window.my || {};

(function(tf, my) {
  class Module {
    constructor() {
      this._params = null;
    }

    async init(paramsDir) {
      // Load parameters
      this.dispose();
      const manifest = await fetch(`${paramsDir}/weights_manifest.json`);
      const manifestJson = await manifest.json();
      this._params = await tf.io.loadWeights(manifestJson, paramsDir);
    }

    dispose() {
      // Dispose of parameters
      if (this._params !== null) {
        for (const n in this._params) {
          this._params[n].dispose();
        }
        this._params = null;
      }
    }
  }

  class LSTMHiddenState {
    constructor(c, h) {
      if (c.length !== h.length) throw "Invalid shapes";
      this.c = c;
      this.h = h;
    }

    dispose() {
      for (let i = 0; i < this.c.length; ++i) {
        this.c[i].dispose();
        this.h[i].dispose();
      }
    }
  }

  function pyTorchLSTMCellFactory(
    kernelInputHidden,
    kernelHiddenHidden,
    biasInputHidden,
    biasHiddenHidden
  ) {
    // Patch between differences in LSTM APIs for PyTorch/Tensorflow
    // NOTE: Fixes kernel packing order
    // PyTorch packs kernel as [i, f, j, o] and Tensorflow [i, j, f, o]
    // References:
    // https://github.com/tensorflow/tfjs/blob/31fd388daab4b21c96b2cb73c098456e88790321/tfjs-core/src/ops/basic_lstm_cell.ts#L47-L78
    // https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM

    // Pack kernel
    const kernel = tf.transpose(
      tf.concat([kernelInputHidden, kernelHiddenHidden], 1)
    );

    // Pack bias
    // NOTE: Not sure why PyTorch breaks bias into two terms...
    const bias = tf.add(biasInputHidden, biasHiddenHidden);

    // Create empty forgetBias
    const forgetBias = tf.scalar(0, "float32");

    return (data, c, h) => {
      // NOTE: Modified from Tensorflow.JS basicLSTMCell (see first reference)

      const combined = tf.concat([data, h], 1);
      const weighted = tf.matMul(combined, kernel);
      const res = tf.add(weighted, bias);

      // i = input_gate, j = new_input, f = forget_gate, o = output_gate
      const batchSize = res.shape[0];
      const sliceCols = res.shape[1] / 4;
      const sliceSize = [batchSize, sliceCols];
      const i = tf.slice(res, [0, 0], sliceSize);
      //const j = tf.slice(res, [0, sliceCols], sliceSize);
      //const f = tf.slice(res, [0, sliceCols * 2], sliceSize);
      const f = tf.slice(res, [0, sliceCols], sliceSize);
      const j = tf.slice(res, [0, sliceCols * 2], sliceSize);
      const o = tf.slice(res, [0, sliceCols * 3], sliceSize);

      const newC = tf.add(
        tf.mul(tf.sigmoid(i), tf.tanh(j)),
        tf.mul(c, tf.sigmoid(tf.add(forgetBias, f)))
      );
      const newH = tf.mul(tf.tanh(newC), tf.sigmoid(o));
      return [newC, newH];
    };
  }

  const DEFAULT_CKPT_DIR =
    "https://chrisdonahue.com/music-cocreation-tutorial/pretrained";
  const PIANO_NUM_KEYS = 88;

  class PianoGenieDecoder extends Module {
    constructor(rnnDim, rnnNumLayers) {
      super();
      this.rnnDim = rnnDim === undefined ? 128 : rnnDim;
      this.rnnNumLayers = rnnNumLayers === undefined ? 2 : rnnNumLayers;
    }

    async init(paramsDir) {
      await super.init(paramsDir === undefined ? DEFAULT_CKPT_DIR : paramsDir);
    }

    initHidden(batchSize) {
      // NOTE: This allocates memory that must later be freed
      const c = [];
      for (let i = 0; i < this.rnnNumLayers; ++i) {
        c.push(tf.zeros([batchSize, this.rnnDim], "float32"));
      }
      const h = [];
      for (let i = 0; i < this.rnnNumLayers; ++i) {
        h.push(tf.zeros([batchSize, this.rnnDim], "float32"));
      }
      return new LSTMHiddenState(c, h);
    }

    forward(kim1, ti, bi, him1) {
      // NOTE: JavaScript API takes in one timestep per call, i.e., [B] rather than [B, S] as in Python

      // Encode input
      const inputs = [
        tf.oneHot(kim1, PIANO_NUM_KEYS + 1),
        tf.expandDims(ti, 1),
        tf.expandDims(bi, 1)
      ];
      let x = tf.concat(inputs, 1);

      // Project encoded inputs
      x = tf.add(
        tf.matMul(x, this._params["dec.input.weight"], false, true),
        this._params[`dec.input.bias`]
      );

      // Create RNN cell function closures
      const cells = [];
      for (let l = 0; l < this.rnnNumLayers; ++l) {
        cells.push(
          pyTorchLSTMCellFactory(
            this._params[`dec.lstm.weight_ih_l${l}`],
            this._params[`dec.lstm.weight_hh_l${l}`],
            this._params[`dec.lstm.bias_ih_l${l}`],
            this._params[`dec.lstm.bias_hh_l${l}`]
          )
        );
      }

      // Run RNN
      if (him1 === undefined || him1 === null) {
        him1 = this.initHidden(kim1.shape[0]);
      }
      const [hic, hih] = tf.multiRNNCell(cells, x, him1.c, him1.h);
      x = hih[this.rnnNumLayers - 1];
      const hi = new LSTMHiddenState(hic, hih);

      // Compute logits
      const hatki = tf.add(
        tf.matMul(x, this._params["dec.output.weight"], false, true),
        this._params[`dec.output.bias`]
      );

      return [hatki, hi];
    }
  }

  const NUM_BUTTONS = 8;

  class IntegerQuantizer extends Module {
    constructor(numBins) {
      super();
      this.numBins = numBins === undefined ? NUM_BUTTONS : numBins;
    }

    discreteToReal(x) {
      x = tf.cast(x, "float32");
      x = tf.div(x, this.numBins - 1);
      x = tf.sub(tf.mul(x, 2), 1);
      return x;
    }
  }

  const TEST_FIXTURES_URI =
    "https://chrisdonahue.com/music-cocreation-tutorial/test/fixtures.json";

  async function testPianoGenieDecoder() {
    const numBytesBefore = tf.memory().numBytes;

    // Create model
    const quantizer = new IntegerQuantizer();
    const decoder = new PianoGenieDecoder();
    await decoder.init();

    // Fetch test fixtures
    const f = await fetch(TEST_FIXTURES_URI).then(r => r.json());

    // Run test
    let totalErr = 0;
    let him1 = null;
    for (let i = 0; i < 128; ++i) {
      him1 = tf.tidy(() => {
        const kim1 = tf.tensor(f["input_keys"][i], [1], "int32");
        const ti = tf.tensor(f["input_dts"][i], [1], "float32");
        let bi = tf.tensor(f["input_buttons"][i], [1], "float32");
        bi = quantizer.discreteToReal(bi);
        const [khati, hi] = decoder.forward(kim1, ti, bi, him1);

        const expectedLogits = tf.tensor(
          f["output_logits"][i],
          [1, 88],
          "float32"
        );
        const err = tf.sum(tf.abs(tf.sub(khati, expectedLogits))).arraySync();
        totalErr += err;

        if (him1 !== null) him1.dispose();
        return hi;
      });
    }

    // Check equivalence to fixtures
    if (isNaN(totalErr) || totalErr > 0.015) {
      console.log(totalErr);
      throw "Failed test";
    }

    // Check for memory leaks
    him1.dispose();
    decoder.dispose();
    if (tf.memory().numBytes !== numBytesBefore) {
      throw "Memory leak";
    }
    quantizer.dispose();

    console.log("Passed test");
  }

  my.PIANO_NUM_KEYS = PIANO_NUM_KEYS;
  my.NUM_BUTTONS = NUM_BUTTONS;
  my.PianoGenieDecoder = PianoGenieDecoder;
  my.IntegerQuantizer = IntegerQuantizer;
  my.testPianoGenieDecoder = testPianoGenieDecoder;
})(window.tf, window.my);
