window.my = window.my || {};

(function(tone, my) {
  const RUN_TEST = false;
  const LOWEST_PIANO_KEY_MIDI_PITCH = 21;

  async function init() {
    // (Optional) Run test case
    if (RUN_TEST) {
      await my.testPianoGenieDecoder();
    }

    // Initialize Piano Genie
    const pianoGenie = new my.PianoGenie();
    await pianoGenie.init();

    // Retrieve UI buttons
    const buttonEls = document.getElementsByTagName("button");

    // Initialize synthesizer state
    const heldButtonToMidiPitch = new Map();
    const synth = new tone.PolySynth(tone.FMSynth).toDestination();

    // Callback for note onset
    let lastTimeMs = null;
    function noteOnset(button) {
      if (heldButtonToMidiPitch.has(button)) return;

      // Run Piano Genie
      const timeMs = new Date().getTime();
      const key = pianoGenie.press(timeMs / 1000, button);

      // Play note out of synthesizer
      if (tone.context.state !== "running") tone.context.resume();
      const midiPitch = key + LOWEST_PIANO_KEY_MIDI_PITCH;
      synth.triggerAttack(tone.Frequency(midiPitch, "midi").toFrequency());
      heldButtonToMidiPitch.set(button, midiPitch);

      // I/O report in console
      let dt =
        lastTimeMs === null ? my.DELTA_TIME_MAX : (timeMs - lastTimeMs) / 1000;
      dt = dt.toFixed(3);
      const latencyMs = new Date().getTime() - timeMs;
      console.log(
        `⌚ ${dt}, 🔘 ${button} -> 🎹🧞 -> 🎵 ${midiPitch} (in ${latencyMs}ms)`
      );
      lastTimeMs = timeMs;

      // Show UI
      buttonEls[button].setAttribute("active", true);
    }

    // Callback for note offset
    function noteOffset(button) {
      if (!heldButtonToMidiPitch.has(button)) return;
      const midiPitch = heldButtonToMidiPitch.get(button);
      synth.triggerRelease(tone.Frequency(midiPitch, "midi").toFrequency());
      heldButtonToMidiPitch.delete(button);
      buttonEls[button].removeAttribute("active");
    }

    // Bind touch control
    for (let b = 0; b < my.NUM_BUTTONS; ++b) {
      const buttonEl = buttonEls[b];
      const doOnset = evt => {
        noteOnset(b);
      };
      const doOffset = evt => {
        noteOffset(b);
      };
      if ("ontouchstart" in window) {
        buttonEl.ontouchstart = doOnset;
        buttonEl.ontouchend = doOffset;
        buttonEl.ontouchleave = doOffset;
      } else {
        buttonEl.onmousedown = doOnset;
        buttonEl.onmouseup = doOffset;
        buttonEl.onmouseout = doOffset;
      }
    }

    // Bind keyboard control
    document.onkeydown = evt => {
      const button = evt.keyCode - 49;
      if (button >= 0 && button < my.NUM_BUTTONS) noteOnset(button);
    };
    document.onkeyup = evt => {
      const button = evt.keyCode - 49;
      if (button >= 0 && button < my.NUM_BUTTONS) noteOffset(button);
    };

    // Show UI
    document.getElementById("loading").style.display = "none";
    document.getElementById("loaded").style.display = "block";
  }

  document.addEventListener("DOMContentLoaded", init, false);
})(window.Tone, window.my);
