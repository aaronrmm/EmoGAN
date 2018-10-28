Project Description

[Goal]
The project goal is to implement a procedural music generator that fits an inputted mood, theme, or genre and 

Gap
Nowadays there exist many neural-network-based music generators. Commercial quality generators include AMPer, Google Magenta, IBM Watson Beat, LANDR, AI Music UK, nSynth, and jukeDeck. Most use Long-Short-Term-Memory networks (LSTMs) and append new notes or phrases to a initial note by predicting which note should come next given the the current track as history and large libraries of music as training data. Or understanding is that this method alone does not support maintaining a specific identifying theme throughout the generated piece.

We believe using a GAN model instead will allow the network to classify if a segment of music contains a variation on the input theme or not, and learn to constrain its generated content to the subset that contains this theme. We would like to first create variations of the input melody before using a GAN to generate music around these variations. To our knowledge no AI based music generator produces variations of a melody as a separate intentional step.

Approach
We will be using the state of the art algorithm for varying a melody according to mood, as well as a state of the art algorithm for applying GANs to generate multi-track polyphonic music from a monophonic track. We will implement a pipeline that feeds the generated mood-specific variations into the GANs generator.

Background and Current State of the Art
GANs for music generation
GANs are popular models for generating images, but so far, little research has been done in applying GANs to music generation, and the quality of music generated by GAN models is far worse than music produced by LSTMs. The state of the art for GAN models for music generation are WaveGAN and MuseGAN [1, 2, 3]. MuseGAN [3] is a state-of-the-art generative model for music generation with GANs. It generates a multi-track polyphonic music with multitrack interdependency and harmonic and rhythmic structure. The architecture of MuseGan is modelled to generate music conditioned on the temporal structure of an inputted track. WaveGAN [2] and its variants are the only other models we found that use GANs to generate music. They take advantage of GANs success in generating images by converting training audio into graphical waveforms (with some unavoidable loss). The output is simply a generated image of a waveform that is then converted back into audio.

Procedurally Generating Mood-Specific Variations of a Melody
SentiMozart [4] is a model that generates music based on a mood. It uses a convolutional layer to read the emotion of a human face and then outputs music with that mood using an LSTM. They do not, however, make efforts to maintain a coherent musical structure across moods. In fact, the mapping from mood to music is trained using a different set of music pieces for each mood.

Ramanto & Maulidevi [5] developed a markov chain model that would generate a sequence of notes and then apply variations that impart an input mood upon the track.

Payoff

The product will be a tool that allows designers to quickly generate music around a specified theme, mood, and genre. The benefits of procedural music generators is that they require less labor than manually creating music. They also allow for music to be created dynamically as needed, which is useful for if a designer needs a musical piece to extend for an unspecified amount of time without looping. The additional constraints we are adding to these music generators will improve the control designers have on the music. In offline use cases, this will further reduce labor by decreasing the number of unwanted samples a designer needs to weed through. For online use cases, this increases reliability that the music generated in an online situation will be what the designer intended, and also lets the generator dynamically and intelligently adjust to changes in the situation the music is accompanying, such as the death of a loved character in a video game. In addition, we hope to further research into the under-explored field of GANs generated music.


Work Plan

Algorithm and Method
We plan to implement a system to meet our goal in three phases. Our first object is to study and implement a system than can alter parts of a melody based on a mood. We will be following the paper on Procedural Music Generator with User Chosen Mood Compatibility [5] for this purpose. The paper achieves this objective by focusing on three goals: understand emotions and their classification, compose music according to a theory or a widely acceptable convention and to devise a way to procedurally compose the music while still adhering to said theory.

Our second objective is to have a working model of museGAN locally. This phase requires replicating the authors’ results using the code provided in their github. We would then generate the piano roll music representation of the melody outputted from the Markov chain in our third phase. We would then require creating the pipeline between these algorithms, tuning hyperparameters of the algorithms, and possibly augmenting the data (libraries of midi music files), as well as understand enough of the algorithms to make any modifications we need to.

Both the algorithms report a degree on noise in the music they generated. Since museGAN produces polyphonic music that can be conditioned on a track, we might face two issues. First, the melody we generate based on the mood may have noise which may affect temporal characteristics that museGAN uses to generate the multi-track music. This may amplify any noise the markov chain model might generate. Second, Ramanto & Maulidevi [5] mention that tempo, pitch and chord type dominance together are manipulated to generate music based on a mood whereas museGAN only uses the conditioning track for temporal interdependency between the tracks. We might not be able to generate multi-track polyphonic music that is in congruence with the user specified mood. In order to account for these two potential risks, our backup plan would be to enhance the markov chain based procedural music generator built by Ramanto & Maulidevi.

Tools
To collaboratively develop this implementation, we will be using github to store and distribute (to the public and each other) our code. MuseGan’s code-base is written in Python, while Ramanto & Maulidevi’s Markov Chain model is written in Java. Because MuseGAN is the more complicated between the two algorithms, we will be using Python so that we can reuse as much of that code as possible. We will have to reimplement the Markov Chain in Python. We will be using the Prett_midi library for Python to process the training files, as well as to convert our output to a playable audio format.

Deliverables
Our deliverables include:
The code to our tool
A link to the training data 
An example set of audio files generated by our tool
The data collected from our user study
Documentation and Readme for running the tool from code

Division of labor
Aaron
Reproduce museGAN results
Create pipeline between algorithms
Create Survey
Develop the documentation

Mehak
Implement markov chain based procedural music generator based on mood
Create pipeline between algorithms
Gather participants for the survey
Develop the documentation


Timeline



Week
Task
1-2
Reproduce MuseGAN, train it and feed it melodies
1-2
Implement markov chain based procedural music generator with user chosen mood compatibility
3
Find and select music dataset for the chosen mood
4
Create pipeline between algorithms
5
Create pipeline between algorithms OR Start implementing backup
6
Evaluate performance
7
Evaluate performance
8
Documentation and additional moods if time permits


Evaluation
We will conduct a user study where we will present users with a set of music tracks, some (20-80%) tracks generated by our algorithm, and some from our training sets. For each track, users will be asked to identify the mood induced, what they believe the base melody to be, if they enjoyed the music, and whether they believe it was generated by a machine.

Rubric
Main Plan:
D: We study any modification of MuseGAN
OR
We study any modification of the markov chain mood generator
C: We implement the markov chain based music generator with user chosen mood capability
B: We attach any melody generator to MuseGAN
B+: We modify and attach any melody generator to MuseGAN 
A: We establish a pipeline between the markov chain based music generator and museGAN to generate music based on a user specified mood
A+:  We attach an extra critic or discriminator to MuseGAN
OR
We support an additional distinct mood and have good results.
OR
We create an interface that allows the user to dynamically switch between moods and or genres as the music is being played

Backup Plan
D: We study any modification of the markov chain mood generator
C: We implement the markov chain based music generator with user chosen mood capability
B: Implement a modification the model: eg. change the value of the transition matrix (their study only uses a convention in music composition), study the use of different instruments, etc.
A: Implement and study the effect of 2 modifications to the model 
A+:  Find a modification that can lead to improvement in the existing model

References
[1] Donahue, C., McAuley, J., & Puckette, M. (2018). Synthesizing Audio with Generative Adversarial Networks. arXiv preprint arXiv:1802.04208.
[2] Lee, C. Y., Toffy, A., Jung, G. J., & Han, W. J. (2018). Conditional WaveGAN. arXiv preprint arXiv:1809.10636.
[3] Dong, H. W., Hsiao, W. Y., Yang, L. C., & Yang, Y. H. (2018). MuseGAN: Multi-track sequential generative adversarial networks for symbolic music generation and accompaniment. In Proc. AAAI Conf. Artificial Intelligence.
[4] Madhok, R., Goel, S., & Garg, S. (2018). SentiMozart: Music Generation based on Emotions.
[5] Ramanto, A. S., & Maulidevi, N. U. (2017). Markov Chain Based Procedural Music Generator with User Chosen Mood Compatibility. International Journal of Asia Digital Art and Design Association, 21(1), 19-24.



[MuseGAN](https://salu133445.github.io/musegan/) is a project on music
generation. In a nutshell, we aim to generate polyphonic music of multiple
tracks (instruments). The proposed models are able to generate music either from
scratch, or by accompanying a track given a priori by the user.

We train the model with training data collected from
[Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)
to generate pop song phrases consisting of bass, drums, guitar, piano and
strings tracks.

Sample results are available
[here](https://salu133445.github.io/musegan/results).

## Prerequisites

> __Below we assume the working directory is the repository root.__

### Install dependencies

- Using pipenv (recommended)

  > Make sure `pipenv` is installed. (If not, simply run `pip install pipenv`.)

  ```sh
  # Install the dependencies
  pipenv install
  # Activate the virtual environment
  pipenv shell
  ```

- Using pip

  ```sh
  # Install the dependencies
  pip install -r requirements.txt
  ```

### Prepare training data

> The training data is collected from
[Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)
(LPD), a new multitrack pianoroll dataset.

```sh
# Download the training data
./scripts/download_data.sh
# Store the training data to shared memory
./scripts/process_data.sh
```

You can also download the training data manually
([train_x_lpd_5_phr.npz](https://docs.google.com/uc?export=download&id=12Z440hxJSGCIhCSYaX5tbvsQA61WD_RH)).

## Scripts

We provide several shell scripts for easy managing the experiments. (See
[here](scripts/README.md) for a detailed documentation.)

> __Below we assume the working directory is the repository root.__

### Train a new model

1. Run the following command to set up a new experiment with default settings.

   ```sh
   # Set up a new experiment
   ./scripts/setup_exp.sh "./exp/my_experiment/" "Some notes on my experiment"
   ```

2. Modify the configuration and model parameter files for experimental settings.

3. You can either train the model:

     ```sh
     # Train the model
     ./scripts/run_train.sh "./exp/my_experiment/" "0"
     ```
   or run the experiment (training + inference + interpolation):

     ```sh
     # Run the experiment
     ./scripts/run_exp.sh "./exp/my_experiment/" "0"
     ```

### Use pretrained models

1. Download pretrained models

   ```sh
   # Download the pretrained models
   ./scripts/download_pretrained_models.sh
   ```

   You can also download the pretrained models manually
   ([pretrained_models.tar.gz](https://docs.google.com/uc?export=download&id=17qJ6jDElLMukwQBZjDEnJctpkyDsd09g)).

2. You can either perform inference from a trained model:

   ```sh
   # Run inference from a pretrained model
   ./scripts/run_inference.sh "./exp/default/" "0"
   ```

   or perform interpolation from a trained model:

   ```sh
   # Run interpolation from a pretrained model
   ./scripts/run_interpolation.sh "./exp/default/" "0"
   ```

## Sample Results

Some sample results can be found in `./exp/` directory. More samples can be
downloaded from the following links.

- [`sample_results.tar.gz`](https://docs.google.com/uc?export=download&id=1OUWv581V9hWPiPGb_amXBdJX-_qoNDi9) (54.7 MB):
  sample inference and interpolation results
- [`training_samples.tar.gz`](https://docs.google.com/uc?export=download&id=1sr68zXGUrX-eC9FGga_Kl58YxZ5R2bc4) (18.7 MB):
  sample generated results at different steps

## Papers

__Convolutional Generative Adversarial Networks with Binary Neurons for
Polyphonic Music Generation__<br>
Hao-Wen Dong and Yi-Hsuan Yang<br>
in _Proceedings of the 19th International Society for Music Information
Retrieval Conference_ (ISMIR), 2018.<br>
[[website](https://salu133445.github.io/bmusegan)]
[[arxiv](https://arxiv.org/abs/1804.09399)]
[[paper](https://salu133445.github.io/bmusegan/pdf/bmusegan-ismir2018-paper.pdf)]
[[slides(long)](https://salu133445.github.io/bmusegan/pdf/bmusegan-tmacw2018-slides.pdf)]
[[slides(short)](https://salu133445.github.io/bmusegan/pdf/bmusegan-ismir2018-slides.pdf)]
[[poster](https://salu133445.github.io/bmusegan/pdf/bmusegan-ismir2018-poster.pdf)]
[[code](https://github.com/salu133445/bmusegan)]

__MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic
Music Generation and Accompaniment__<br>
Hao-Wen Dong\*, Wen-Yi Hsiao\*, Li-Chia Yang and Yi-Hsuan Yang,
(\*equal contribution)<br>
in _Proceedings of the 32nd AAAI Conference on Artificial Intelligence_
(AAAI), 2018.<br>
[[website](https://salu133445.github.io/musegan)]
[[arxiv](http://arxiv.org/abs/1709.06298)]
[[paper](https://salu133445.github.io/musegan/pdf/musegan-aaai2018-paper.pdf)]
[[slides](https://salu133445.github.io/musegan/pdf/musegan-aaai2018-slides.pdf)]
[[code](https://github.com/salu133445/musegan)]

__MuseGAN: Demonstration of a Convolutional GAN Based Model for Generating
Multi-track Piano-rolls__<br>
Hao-Wen Dong\*, Wen-Yi Hsiao\*, Li-Chia Yang and Yi-Hsuan Yang
(\*equal contribution)<br>
in _ISMIR Late-Breaking Demos Session_, 2017.
(non-refereed two-page extended abstract)<br>
[[paper](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf)]
[[poster](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-poster.pdf)]
