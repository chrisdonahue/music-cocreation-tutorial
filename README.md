## Interactive music co-creation with PyTorch and TensorFlow.js

**NOTE:** This tutorial is in-progress but will be finalized by November 7th, 2021

This tutorial is a start-to-finish demonstration ([click here for result](https://chrisdonahue.com/music-cocreation-tutorial)) of building an interactive music co-creation system in two parts:

1. <a href="https://colab.research.google.com/drive/124pk1yehPx1y-K3hBG6-SoUSVqQ-RWnM?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> **Training a generative model of music in Python (via PyTorch)**
2. <a href="https://glitch.com/edit/#!/music-cocreation-tutorial" target="_blank"><img src="https://cdn.glitch.com/2703baf2-b643-4da7-ab91-7ee2a2d00b5b%2Fremix-button.svg" alt="Remix on Glitch"/></a>**Deploying it in JavaScript (via TensorFlow.js)**

This demonstration was prepared by [Chris Donahue](https://chrisdonahue.com) as part of an [ISMIR 2021 tutorial](https://ismir2021.ismir.net/tutorials/) on *Designing generative models for interactive co-creation*, co-hosted by [Anna Huang](https://research.google/people/105787/) and [Jon Gillick](https://www.jongillick.com/).

The example generative model we will train and deploy is [Piano Genie](https://magenta.tensorflow.org/pianogenie) (Donahue et al. 2019). Piano Genie allows anyone to improvise on the piano by mapping performances on a miniature 8-button keyboard to realistic performances on a full 88-key piano in real time. At a low-level, it is an LSTM that operates on symbolic music data (i.e., MIDI), and is lightweight enough for real-time performance on mobile CPUs.

### Part 1: Training in Python

<a href="https://colab.research.google.com/drive/124pk1yehPx1y-K3hBG6-SoUSVqQ-RWnM?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This part of the tutorial involves training the Piano Genie generative model from scratch in PyTorch, which comes in the form of a self-contained [Google Colab notebook](https://colab.research.google.com/drive/124pk1yehPx1y-K3hBG6-SoUSVqQ-RWnM?usp=sharing). The instructions for this part are embedded in the Colab, and the model takes about an hour to train on Colab's free GPUs.

The outputs of this part are: (1) a [model checkpoint](part-2-js-interaction/pretrained), and (2) [serialized inputs and outputs for a test case](part-2-js-interaction/test/fixtures.json), which we will use to check correctness of our JavaScript port in the next part.

### Part 2: Deploying in JavaScript

<a href="https://glitch.com/edit/#!/music-cocreation-tutorial" target="_blank"><img src="https://cdn.glitch.com/2703baf2-b643-4da7-ab91-7ee2a2d00b5b%2Fremix-button.svg" alt="Remix on Glitch"/></a>

This part of the tutorial involves porting the Piano Genie decoder to TensorFlow.js, and hooking the model up to a simple UI to allow users to interact with the model. The final result lives at [`part-2-js-interaction`](part-2-js-interaction), and is [hosted here](https://chrisdonahue.com/music-cocreation-tutorial).

#### Redefining the model in TensorFlow.js

#### Testing for correctness

#### Hooking model up to simple UI

### End matter

#### Licensing info

This tutorial uses the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) (Hawthorne et al. 2018), which is distributed under a [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/). Because of the ShareAlike clause, the material in this tutorial is also be distributed under that same license.

#### Acknowledgements

Thanks to Ian Simon and Sander Dieleman, co-authors on Piano Genie, and to Monica Dinculescu for creating the original [Piano Genie demo](https://piano-genie.glitch.me).

#### Attribution

If this tutorial was useful to you, please consider citing this repository.

```
@software{donahue2021tutorial,
  author={Donahue, Chris and Huang, Cheng-Zhi Anna and Gillick, Jon},
  title={Interactive music co-creation with {PyTorch} and {TensorFlow.js}},
  url={https://github.com/chrisdonahue/music-cocreation-tutorial},
  year={2021}
}
```

If Piano Genie was useful to you, please consider citing our original paper.

```
@inproceedings{donahue2019genie,
  title={Piano Genie},
  author={Donahue, Chris and Simon, Ian and Dieleman, Sander},
  booktitle={Proceedings of the 24th ACM Conference on Intelligent User Interfaces},
  year={2019}
}
```
