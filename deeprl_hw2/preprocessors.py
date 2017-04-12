"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self):
        pass

    def process_frame_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        
        assert state.ndim == 3  # (height, width, channel)
        
        img = Image.fromarray(state)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_state = np.array(img)
        assert processed_state.shape == INPUT_SHAPE
        return processed_state.astype('uint8')  # saves storage in experience memory
        
    def process_frame_for_network(self, frame):
        """Scale, convert to greyscale and store as float32.
            
            Basically same as process state for memory, but this time
            outputs float32 images.
            """
        processed_frame = state.astype('float32') / 255.
        return processed_frame

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.
            state is a list of frames (the oldest frames
            are at its beginning). It returns an np.array
            
        Basically same as process state for memory, but this time
        outputs float32 images. state is a list of frames
        """

        processed_state=np.array(state)
        #print processed_state.shape
        assert processed_state.shape==(WINDOW_LENGTH,INPUT_SHAPE[0],INPUT_SHAPE[1])
        
        processed_state = processed_state.astype('float32') / 255.
        
    
        return processed_state

    def process_samples_for_network(self, samples):
        """samples is a ringbuffer which contains the 4 most recent states
            create a processed array that will feed the network
            the most recent state is the last entry in the 3d array.
        """
        #create a list which contains the most recent frame at the right
        state=[]
        for idx in range(0,samples.length):
            state.insert(0,samples[samples.length-1-idx])
        
        while len(state) < WINDOW_LENGTH:
            state.insert(0, np.zeros(state[0].shape))
        
        assert len(state)==WINDOW_LENGTH
        #assert not (np.array_equal(state[0],state[1]))  or (np.array_equal(state[0],np.zeros(state[0].shape)))
        #assert not (np.array_equal(state[1],state[2]))  or (np.array_equal(state[1],np.zeros(state[0].shape)))
        #assert not (np.array_equal(state[2],state[3]))  or (np.array_equal(state[2],np.zeros(state[0].shape)))
        
        processed_state=self.process_state_for_network(state)
        #print processed_state
        return processed_state

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        
        return np.clip(reward, -1., 1.)




