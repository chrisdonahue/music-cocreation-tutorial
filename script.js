window.my = window.my || {};

(function(tone, my) {
  const RUN_TEST = false;
  const NUM_BUTTONS = 8;
  const LOWEST_PIANO_KEY_MIDI_NOTE = 21;

  async function init() {
    // (Optional) Run test case
    if (RUN_TEST) {
      await my.testPianoGenie();
    }

    // Initialize Piano Genie
    const pianoGenie = new my.PianoGenie();
    await pianoGenie.init();

    // Retrieve UI buttons
    const buttonEls = document.getElementsByTagName("button");

    // Initialize synthesizer state
    let lastTimeMs = new Date().getTime();
    const heldButtonToMidiNote = new Map();
    const synth = new tone.PolySynth(tone.FMSynth).toDestination();

    // Callback for note onset
    function noteOnset(button) {
      if (heldButtonToMidiNote.has(button)) return;

      // Run Piano Genie
      const timeMs = new Date().getTime();
      const key = pianoGenie.press(timeMs / 1000, button);

      // Play note out of synthesizer
      if (tone.context.state !== "running") tone.context.resume();
      const midiPitch = key + LOWEST_PIANO_KEY_MIDI_NOTE;
      synth.triggerAttack(tone.Frequency(midiPitch, "midi").toFrequency());
      heldButtonToMidiNote.set(button, midiPitch);

      // Show UI
      buttonEls[button].setAttribute("active", true);

      // I/O report in console
      const dt = ((timeMs - lastTimeMs) / 1000).toFixed(3);
      const latencyMs = new Date().getTime() - timeMs;
      console.log(
        `⌚ ${dt}, 🔘 ${button} -> 🎹🧞 -> 🎵 ${midiPitch} (in ${latencyMs}ms)`
      );
      lastTimeMs = timeMs;
    }

    // Callback for note offset
    function noteOffset(button) {
      if (!heldButtonToMidiNote.has(button)) return;
      const midiPitch = heldButtonToMidiNote.get(button);
      synth.triggerRelease(tone.Frequency(midiPitch, "midi").toFrequency());
      heldButtonToMidiNote.delete(button);
      buttonEls[button].removeAttribute("active");
    }

    // Bind touch control
    for (let b = 0; b < NUM_BUTTONS; ++b) {
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
      if (button >= 0 && button < NUM_BUTTONS) noteOnset(button);
    };
    document.onkeyup = evt => {
      const button = evt.keyCode - 49;
      if (button >= 0 && button < NUM_BUTTONS) noteOffset(button);
    };

    // Show UI
    document.getElementById("loading").style.display = "none";
    document.getElementById("loaded").style.display = "block";
  }

  document.addEventListener("DOMContentLoaded", init, false);
})(window.Tone, window.my);
