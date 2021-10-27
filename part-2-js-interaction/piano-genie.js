(function(tf, my) {
  const DELTA_TIME_MAX = 1;
  const SOS = my.PIANO_NUM_KEYS;
  const DEFAULT_TEMPERATURE = 0.25;

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

  class PianoGenie {
    constructor() {
      // Model
      this.dec = new my.PianoGenieDecoder();
      this.quant = new my.IntegerQuantizer();

      // Performance state
      this.lastTime = null;
      this.lastKey = null;
      this.lastHidden = null;
    }

    async init() {
      await this.dec.init();

      // Warm start
      this.press(0, 0);
      this.reset();
    }

    reset() {
      if (this.lastHidden !== null) {
        this.lastHidden.dispose();
      }
      this.lastTime = null;
      this.lastKey = null;
      this.lastHidden = null;
    }

    dispose() {
      if (this.lastHidden !== null) {
        this.lastHidden.dispose();
      }
      this.dec.dispose();
    }

    press(time, button, temperature) {
      // Check inputs
      temperature =
        temperature === undefined ? DEFAULT_TEMPERATURE : temperature;
      let deltaTime = this.lastTime === null ? 1e6 : time - this.lastTime;
      if (deltaTime < 0) {
        console.log("Warning: Specified time is in the past");
        deltaTime = 0;
      }
      if (deltaTime > DELTA_TIME_MAX) {
        deltaTime = DELTA_TIME_MAX;
      }
      if (button < 0 || button >= my.NUM_BUTTONS) {
        throw "Specified button is out of range";
      }

      // Run model
      const lastKey = this.lastKey === null ? SOS : this.lastKey;
      const [key, hidden] = tf.tidy(() => {
        const kim1 = tf.tensor(lastKey, [1], "int32");
        let ti = tf.tensor(deltaTime, [1], "float32");
        let bi = tf.tensor(button, [1], "int32");
        bi = this.quant.discreteToReal(bi);
        const him1 = this.lastHidden;
        const [hatki, hi] = this.dec.forward(kim1, ti, bi, him1);
        const ki = sampleFromLogits(tf.squeeze(hatki), temperature);
        return [ki.dataSync()[0], hi];
      });

      // Update state
      this.lastTime = time;
      this.lastKey = key;
      if (this.lastHidden !== null) this.lastHidden.dispose();
      this.lastHidden = hidden;

      return key;
    }
  }

  my.DELTA_TIME_MAX = DELTA_TIME_MAX;
  my.PianoGenie = PianoGenie;
})(window.tf, window.my);
