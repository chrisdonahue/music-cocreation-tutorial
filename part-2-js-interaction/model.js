window.my = window.my || {};

(function(tf, my) {
  class LSTMHiddenState {
    constructor(batchSize, numLayers, dim, c, h) {
      this.batchSize = batchSize;
      this._numLayers = numLayers;
      this._dim = dim;

      if (c === undefined) {
        c = [];
        for (let i = 0; i < this._numLayers; ++i) {
          c.push(tf.zeros([this.batchSize, this._dim], "float32"));
        }
      }
      this.c = c;

      if (h === undefined) {
        h = [];
        for (let i = 0; i < this._numLayers; ++i) {
          h.push(tf.zeros([this.batchSize, this._dim], "float32"));
        }
      }
      this.h = h;
    }

    dispose() {
      for (let i = 0; i < this._numLayers; ++i) {
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

  function sampleFromLogits(logits, temperature, seed) {
    temperature = temperature !== undefined ? temperature : 1;
    if (temperature < 0 || temperature > 1) {
      throw "Specified invalid temperature";
    }

    let result;
    if (temperature === 0) {
      result = tf.argMax(logits, 0);
    } else {
      if (temperature < 1) {
        logits = tf.div(logits, tf.scalar(temperature, "float32"));
      }
      const scores = tf.reshape(tf.softmax(logits, 0), [1, -1]);
      const sample = tf.multinomial(scores, 1, seed, true);
      result = tf.reshape(sample, []);
    }

    return result;
  }

  const DEFAULT_CKPT_DIR =
    "https://chrisdonahue.com/music-cocreation-tutorial/pretrained";
  const DEFAULT_TEMPERATURE = 0.25;
  const PIANO_NUM_KEYS = 88;

  class PianoGenie {
    constructor(numButtons, deltaTimeMax, rnnNumLayers, rnnDim) {
      this._params = null;

      // Model config
      this.numButtons = numButtons === undefined ? 8 : numButtons;
      this.deltaTimeMax = deltaTimeMax === undefined ? 1 : deltaTimeMax;
      this.rnnNumLayers = rnnNumLayers === undefined ? 2 : rnnNumLayers;
      this.rnnDim = rnnDim === undefined ? 128 : rnnDim;

      // Performance state
      this.time = null;
      this.lastKey = null;
      this.hidden = null;
    }

    reset() {
      if (this.hidden !== null) {
        this.hidden.dispose();
      }
      this.time = null;
      this.lastKey = PIANO_NUM_KEYS;
      this.hidden = this.initHidden(1);
    }

    async init(paramsDir) {
      this.dispose();

      // Load params
      paramsDir = paramsDir === undefined ? DEFAULT_CKPT_DIR : paramsDir;
      const manifest = await fetch(`${paramsDir}/weights_manifest.json`);
      const manifestJson = await manifest.json();
      this._params = await tf.io.loadWeights(manifestJson, paramsDir);

      // Warm start
      this.reset();
      this.press(0, 0);
      this.reset();
    }

    dispose() {
      // Dispose params
      if (this._params !== null) {
        for (const n in this._params) {
          this._params[n].dispose();
        }
        this._params = null;
      }

      // Dispose hidden state
      if (this.hidden !== null) {
        this.hidden.dispose();
      }
    }

    press(time, button, temperature) {
      if (this._params === null) {
        throw "Call initialize first";
      }

      // Check inputs
      temperature =
        temperature === undefined ? DEFAULT_TEMPERATURE : temperature;
      const deltaTime = this.time === null ? 1e6 : time - this.time;
      if (deltaTime < 0) {
        console.log("Warning: Specified time is in the past");
        deltaTime = 0;
      }
      if (button < 0 || button >= this.numButtons) {
        throw "Specified button is out of range";
      }

      // Run model
      const [key, h] = tf.tidy(() => {
        const [logits, h] = this.forward(
          tf.tensor(deltaTime, [1], "float32"),
          tf.tensor(this.lastKey, [1], "int32"),
          tf.tensor(button, [1], "int32"),
          this.hidden
        );
        const key = sampleFromLogits(
          tf.squeeze(logits),
          temperature
        ).dataSync()[0];
        return [key, h];
      });

      // Update state
      this.time = time;
      this.lastKey = key;
      this.hidden.dispose();
      this.hidden = h;

      return key;
    }

    initHidden(batchSize) {
      return new LSTMHiddenState(batchSize, this.rnnNumLayers, this.rnnDim);
    }

    forward(deltaTime, lastKey, button, h) {
      const batchSize = deltaTime.shape[0];

      // Encode inputs
      deltaTime = tf.minimum(deltaTime, this.deltaTimeMax);
      button = tf.sub(
        tf.mul(tf.div(tf.cast(button, "float32"), this.numButtons - 1), 2),
        1
      );
      let x = tf.concat(
        [
          tf.expandDims(deltaTime, 1),
          tf.oneHot(lastKey, PIANO_NUM_KEYS + 1),
          tf.expandDims(button, 1)
        ],
        1
      );

      // Project inputs
      x = tf.add(
        tf.matMul(x, this._params["dec.input.weight"], false, true),
        this._params[`dec.input.bias`]
      );

      // Create RNN cell functions
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
      const [cUpdated, hUpdated] = tf.multiRNNCell(cells, x, h.c, h.h);
      x = hUpdated[this.rnnNumLayers - 1];
      h = new LSTMHiddenState(
        batchSize,
        this.rnnNumLayers,
        this.rnnDim,
        cUpdated,
        hUpdated
      );

      // Compute logits
      x = tf.add(
        tf.matMul(x, this._params["dec.output.weight"], false, true),
        this._params[`dec.output.bias`]
      );

      return [x, h];
    }
  }

  const TEXT_FIXTURES_URI =
    "https://chrisdonahue.com/music-cocreation-tutorial/test/fixtures.json";

  async function testPianoGenie() {
    const numBytesBefore = tf.memory().numBytes;

    // Create model
    const pianoGenie = new PianoGenie();
    await pianoGenie.init();

    // Fetch test fixtures
    const f = await fetch(TEXT_FIXTURES_URI).then(r => r.json());

    // Run test
    let totalErr = 0;
    tf.tidy(() => {
      let logits;
      let h = pianoGenie.initHidden(1);
      for (let i = 0; i < 128; ++i) {
        const dt = tf.tensor(f["input_dts"][i], [1], "float32");
        const key = tf.tensor(f["input_keys"][i], [1], "int32");
        const button = tf.tensor(f["input_buttons"][i], [1], "float32");
        [logits, h] = pianoGenie.forward(dt, key, button, h);

        const expectedLogits = tf.tensor(
          f["output_logits"][i],
          [1, 88],
          "float32"
        );
        const err = tf.sum(tf.abs(tf.sub(logits, expectedLogits))).arraySync();
        totalErr += err;
      }
    });

    // Check equivalence to fixtures
    if (totalErr > 0.015) {
      console.log(totalErr);
      throw "Failed test";
    }

    // Check for memory leaks
    pianoGenie.dispose();
    if (tf.memory().numBytes !== numBytesBefore) {
      throw "Memory leak";
    }

    console.log("Passed test");
  }

  my.PianoGenie = PianoGenie;
  my.testPianoGenie = testPianoGenie;
})(window.tf, window.my);
