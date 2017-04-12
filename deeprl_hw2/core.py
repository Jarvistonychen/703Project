"""Core classes."""
from collections import namedtuple
from PIL import Image
import random
import numpy as np

Sample = namedtuple('Experience', 'state, action, reward, next_state, terminal')

DEBUG=0
class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass

class RingBuffer(object):
    
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]
    

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]
    def __str__(self):
        return 'start pointer {0} length {1} max length {2}'.format(self.start, self.length, self.maxlen)
        
    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    



    def __init__(self, max_size, window_length):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        
        self.max_size=max_size
        self.window_length=window_length
        
        self.actions = RingBuffer(max_size)
        self.rewards = RingBuffer(max_size)
        self.terminal = RingBuffer(max_size)
        self.observations = RingBuffer(max_size)
        
        
        pass

    @property
    def nb_entries(self):
        return self.observations.length

    def append(self, state, action, reward,is_terminal, flickering=False):
        if flickering==True:
            flicker=np.random.binomial(1,0.5)
        
            if flicker:
                self.observations.append(np.uint8(np.zeros(state.shape)))
            else:
                self.observations.append(state)
        else:
            self.observations.append(state)
        
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminal.append(is_terminal)
    
        if DEBUG:
            print self.observations
            print self.actions
            print self.rewards
            print self.terminal


    def sample(self, batch_size, indexes=None):
        
        if indexes is None:
        #if indexes is None draw random samples, each index refers to the last frame of next_state
        #print self.nb_entries
        #    print batch_size
            indexes = random.sample(xrange(2, self.nb_entries), batch_size)
      
      
        assert 2<= np.min(indexes)< self.nb_entries
        
        assert len(indexes) == batch_size
            
        # state = [idx-wl, idx-wl+1, idx-wl+2,...,idx-1], wl=window length
        # next_state = [idx-wl+1,.....,idx]
        # make sure idx_1 is not terminal so that state next state belong to the same episode
        # if one of the frames <idx-1 is terminal just zero it
        
        samples = []

        for idx in indexes:
            
            #the last frame (at idx-1) of state is terminal (indicated by observation at idx-2), new episode begins for last frame of next_state (at idx)
            terminal=self.terminal[idx-2]
            while terminal:
                idx= random.sample(xrange(2,self.nb_entries),1)
                idx=idx[0]
                
                assert 2<=idx<self.nb_entries
                #print idx
                
                terminal=self.terminal[idx-2]

            if DEBUG:
                print 'idx sample {0}'.format(idx)
            #form the state
            state = [self.observations[idx-1]] #we know that this is not the last frame of episode and idx-1>0
            

            
            for offset in range(2,self.window_length+1):
                cur_idx = idx - offset
                
                #if we moved to a frame which was added after the frame at idx, or
                #we can't tell if the current frame is terminal (from its previous frame, which has been overwritten)
                #or if the frame it's terminal
                if cur_idx < 0 or  cur_idx-1 < 0 or self.terminal[cur_idx - 1]:
                    break

                state.insert(0,self.observations[cur_idx])

            #fill-in the state with zeroes in case it is needed

            while len(state) < self.window_length:
                state.insert(0, np.uint8(np.zeros(state[0].shape)))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            next_terminal = self.terminal[idx - 1]
            
            if DEBUG:
                for fr_id in range(0,self.window_length):
                    state_img=Image.fromarray(state[fr_id])
                    state_img.save('sample_{0}_state_{1}.jpg'.format(idx,fr_id))
        
        
            next_state = [np.copy(x) for x in state[1:]]
            next_state.append(self.observations[idx])

            if DEBUG:
                for fr_id in range(0,self.window_length):
                    next_state_img=Image.fromarray(next_state[fr_id])
                    next_state_img.save('sample_{0}_next_state_{1}.jpg'.format(idx,fr_id))
    
            assert len(state) == self.window_length
            assert len(next_state) == len(state)
            
            if DEBUG:
                print 'action{0} reward next{1} terminal{1}'.format(action, reward, next_terminal)
                
            samples.append(Sample(state=state, action=action, reward=reward,
                                          next_state=next_state, terminal=next_terminal))
                
        assert len(samples) == batch_size
        return samples
    

    def clear(self):
        del self.observations
        del self.actions
        del self.rewards
        del self.terminals

        self.close()

