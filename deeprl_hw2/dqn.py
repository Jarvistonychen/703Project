"""Main DQN agent."""

import keras

from PIL import Image
from keras import optimizers
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,Multiply,BatchNormalization)
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import History
from keras.models import Model
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

import deeprl_hw2 as tfrl
from copy import deepcopy

from core import RingBuffer

DEBUG=0
DEBUG_FRAMES=1

GAMMA = 0.99
ALPHA = 25e-5
EPSILON = 0.05
REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 32
IMG_ROWS , IMG_COLS = 84, 84
WINDOW_LENGTH = 4
TARGET_QNET_RESET_INTERVAL = 10000
SAMPLES_BURN_IN = 50000
TRAINING_FREQUENCY=4
NUM_RAND_STATE = 1000
EVALUATION_FREQUENCY=10000
#EVALUATION_FREQUENCY=2
SAVE_FREQUENCY=1000000
EVAL_NUM_EPISODES=20


class QNAgent:
    
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 network_type,
                 agent_type,
                 num_actions,
                 preprocessor,
                 memory,
                 burnin_policy,
                 training_policy,
                 testing_policy,
                 gamma=GAMMA,
                 alpha=ALPHA,
                 target_update_freq=TARGET_QNET_RESET_INTERVAL,
                 num_burn_in=SAMPLES_BURN_IN,
                 train_freq=TRAINING_FREQUENCY,
                 eval_freq=EVALUATION_FREQUENCY,
                 batch_size=BATCH_SIZE):


        self.network_type 	= network_type
        self.agent_type     = agent_type
        self.num_actions	= num_actions
        self.atari_proc  	= preprocessor
     

   

        self.memory	 	= memory
        self.burnin_policy      = burnin_policy
        self.testing_policy 	= testing_policy
        self.training_policy 	= training_policy
        self.gamma 		= gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in 	= num_burn_in
        self.train_freq		= train_freq
        self.eval_freq      	= eval_freq
        self.batch_size		= batch_size
        self.num_updates 	= 0
        self.num_samples    	= 0
        self.total_reward  	= []
        self.alpha 		= alpha
        self.eval_states=np.zeros((NUM_RAND_STATE, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))
        self.eval_states_mask 	= np.ones((NUM_RAND_STATE, self.num_actions))
        self.input_dummymask = np.ones((1,self.num_actions))
        self.input_dummymask_batch=np.ones((self.batch_size, self.num_actions))
    
        sess = tf.Session()
        self.writer = tf.summary.FileWriter('./logs_'+agent_type+'_'+network_type, sess.graph)
  
    

    @staticmethod
    def save_scalar(step, name, value, writer):
        """Save a scalar value to tensorboard.
            
            Parameters
            ----------
            step: int
            Training step (sets the position on x-axis of tensorboard graph.
            name: str
            Name of variable. Will be the name of the graph in tensorboard.
            value: float
            The value of the variable at this step.
            writer: tf.FileWriter
            The tensorboard FileWriter instance.
        """
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = float(value)
        summary_value.tag = name
        writer=writer.add_summary(summary, step)

    @staticmethod
    def create_dueling_model(self, window, input_shape, num_actions):  # noqa: D103
        """Create a deep network for the Q-network model.
            Parameters
            ----------
            window: int
            Each input to the network is a sequence of frames. This value
            defines how many frames are in the sequence.
            input_shape: tuple(int, int)
            The expected input image size.
            num_actions: int
            Number of possible actions. Defined by the gym environment.
            model_name: str
            Useful when debugging. Makes the model show up nicer in tensorboard.
            
            Returns
            -------
            keras.models.Model
            The Q-model.
        """
        
        a1 = Input(shape=(window,input_shape[0],input_shape[1]))
        a2 = Input(shape=(num_actions,))
        b = Conv2D(32, (8, 8), strides=4, padding='same',use_bias=True, data_format='channels_first')(a1)
        bn = BatchNormalization(axis=1)(b)
        b = Activation ('relu')(bn)
        c = Conv2D(64, (4, 4), strides=2, padding='same',use_bias=True, data_format='channels_first')(b)
        cn = BatchNormalization(axis=1)(c)
        c = Activation ('relu')(cn)

        d = Conv2D(64, (3, 3), strides=1, padding='same',use_bias=True, data_format='channels_first')(c)
        dn = BatchNormalization(axis=1)(d)
        d = Activation ('relu')(dn)

        d = Flatten()(d)
        e1 = Dense(512)(d)
        e2 = Dense(512)(d)
        e1 = Activation ('relu')(e1)
        e2 = Activation ('relu')(e2)
        f_adv = Dense(num_actions)(e1)
        f_adv = Activation ('linear')(f_adv)
        f_adv_mean = GlobalAveragePooling1D()(f_adv)
        f_val = Dense(1)(e2)
        f_val = Activation ('linear')(f_val)
        f_val_minus_mean = Add()([f_val,-f_adv_mean])
        f_val_minus_mean = RepeatVector(num_actions)(f_val_minus_mean)
        f_val_minus_mean = Flatten()(f_val_minus_mean)
        f = Add()([f_adv, f_val_minus_mean])
        h = Multiply()([f,a2])
        model = Model(inputs=[a1,a2], outputs=[h])
                          
        return model

    @staticmethod
    def create_deep_model(window, input_shape, num_actions):  # noqa: D103
        """Create a deep network for the Q-network model.
            Parameters
            ----------
            window: int
            Each input to the network is a sequence of frames. This value
            defines how many frames are in the sequence.
            input_shape: tuple(int, int)
            The expected input image size.
            num_actions: int
            Number of possible actions. Defined by the gym environment.
            model_name: str
            Useful when debugging. Makes the model show up nicer in tensorboard.
            
            Returns
            -------
            keras.models.Model
            The Q-model.
        """
        
        a1 = Input(shape=(window,input_shape[0],input_shape[1]))
        a2 = Input(shape=(num_actions,))
        b = Conv2D(16, (8, 8), strides=4, padding='same',use_bias=True, data_format='channels_first')(a1)
        b = Activation ('relu')(b)
        c = Conv2D(32, (4, 4), strides=2, padding='same',use_bias=True, data_format='channels_first')(b)
        c = Activation ('relu')(c)
        d = Flatten()(c)
        e = Dense(256)(d)
        e = Activation ('relu')(e)
        f = Dense(num_actions)(e)
        f = Activation ('linear')(f)
        h = Multiply()([f,a2])
        model = Model(inputs=[a1,a2], outputs=[h])
        
        
                          
        return model
    
    @staticmethod
    def create_lstm_deep_model(window, input_shape, num_actions):  # noqa: D103
        """Create a deep network for the Q-network model.
            Parameters
            ----------
            window: int
            Each input to the network is a sequence of frames. This value
            defines how many frames are in the sequence.
            input_shape: tuple(int, int)
            The expected input image size.
            num_actions: int
            Number of possible actions. Defined by the gym environment.
            model_name: str
            Useful when debugging. Makes the model show up nicer in tensorboard.
            
            Returns
            -------
            keras.models.Model
            The Q-model.
            """
        
        #inputs
        a1 = Input(shape=(window,input_shape[0],input_shape[1]))
        a2 = Input(shape=(num_actions,))
        
        #convolutional layers
        b = Conv2D(32, (8, 8), strides=4, padding='same',use_bias=True, data_format='channels_first')(a1)
        b = Activation ('relu')(b)
        c = Conv2D(64, (4, 4), strides=2, padding='same',use_bias=True, data_format='channels_first')(b)
        c = Activation ('relu')(c)
        c = Conv2D(64, (3, 3), strides=1, padding='same',use_bias=True, data_format='channels_first')(b)
        c = Activation ('relu')(c)
        
        
        d = Flatten()(c)
        
        #Recurrent network
        
        #read:
        
        #http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
        #https://keras.io/layers/recurrent/
        
        
        activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0
        model.add(LSTM(64*WINDOW_LENGTH, input_shape=(1, LOOK_BACK)))
        
        
        #e = Dense(256)(d)
        #e = Activation ('relu')(e)
        
        
        f = Dense(num_actions)(e)
        f = Activation ('linear')(f)
        h = Multiply()([f,a2])
        model = Model(inputs=[a1,a2], outputs=[h])
        
        
        
        return model
    
    @staticmethod
    def create_linear_model(window, input_shape, num_actions):  # noqa: D103
        """Create a linear network for the Q-network model.
            
            Parameters
            ----------
            window: int
            Each input to the network is a sequence of frames. This value
            defines how many frames are in the sequence.
            input_shape: tuple(int, int)
            The expected input image size.
            num_actions: int
            Number of possible actions. Defined by the gym environment.
            model_name: str
            Useful when debugging. Makes the model show up nicer in tensorboard.
            
            Returns
            -------
            keras.models.Model
            The Q-model.
            """
                
        a1 = Input(shape=(window,input_shape[0],input_shape[1]))
        a2 = Input(shape=(num_actions,))
        b = Flatten()(a1)
        c = Dense(num_actions)(b)
        e = Multiply()([c,a2])
        model = Model(inputs=[a1,a2], outputs=[e])
            
        return model


    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        #calculate q-values only for one state
        if(state[0].ndim==3):
            #put the frames of the state in an (4d) array to use the function predict
            assert state[0].shape==(WINDOW_LENGTH,IMG_ROWS,IMG_COLS) or state[0].shape==(IMG_ROWS,IMG_COLS) or state[0].shape==(NUM_RAND_STATE,IMG_ROWS,IMG_COLS)
            assert state[1].shape==(1,self.num_actions)
            batch_state = np.zeros((1,WINDOW_LENGTH, IMG_ROWS,IMG_COLS))
            batch_state[0,:,:,:]=state[0]
            q_values =self.q_network.predict([batch_state,state[1]],batch_size=1)
            
            assert q_values.shape==(1,self.num_actions)
            
            QNAgent.save_scalar(self.num_updates, 'mean q training value', np.mean(q_values) , self.writer)
    
        #calculate q-values for a batch of states
        else:
            q_values =self.q_network.predict(state,batch_size=1)
            assert q_values.shape==(state[0].shape[0],self.num_actions)
        
        return q_values


    def select_action(self, policy,**kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
    

        if policy == 'training':
                if 'state' in kwargs:
                    state=kwargs['state']
                    assert len(state)==2
                    assert state[0].shape==(WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
                    assert state[1].shape==(1,self.num_actions)
                    action=self.training_policy.select_action(self.calc_q_values(kwargs['state']), self.num_samples)
                else:
                    return self.training_policy.select_action()
        elif policy == 'testing':
                if 'state' in kwargs:
                    state=kwargs['state']
                    assert len(state)==2
                    assert state[0].shape==(WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
                    assert state[1].shape==(1,self.num_actions)
                    
                    action=self.testing_policy.select_action(self.calc_q_values(kwargs['state']))
                else:
                    return self.testing_policy.select_action()
        elif policy == 'burnin':
                if 'state' in kwargs:
                    state=kwargs['state']
                    assert len(state)==2
                    assert state[0].shape==(WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
                    assert state[1].shape==(1,self.num_actions)
                    action=self.burnin_policy.select_action(self.calc_q_values(kwargs['state']))
                else:
                    return self.burnin_policy.select_action()

        assert 0<=action<self.num_actions
        return action
            
    def fit(self, env, eval_env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        
        
        #burn-in the replay memory
        next_state=env.reset()
        assert next_state is not None

        for step in range(self.num_burn_in):
            
            state = next_state
            action = self.select_action(policy='burnin')
            next_state, reward, is_terminal, debug_info = env.step(action)
        
            
            mem_proc_state=self.atari_proc.process_frame_for_memory(state)
            assert mem_proc_state is not None
            assert mem_proc_state.shape == (IMG_ROWS,IMG_COLS)
            
            #apply action while being on state, receive reward move to next state which is_terminal
            self.memory.append(mem_proc_state, \
                               action, \
                               reward, \
                               is_terminal, \
                               True)
                               
            self.num_samples += 1
        
        #keep some encounteered states during the burn-in for evaluation
        if DEBUG:
            print '=========== Memory burn in ({0}) finished =========='.format(self.num_burn_in)
        
    
  
        training_total_reward =0
        training_num_episodes =0
        
        while self.num_samples < num_iterations:
        
            state = env.reset()
         
            
            #buffer which contains the WINDOW_LENGTH most recent observations
            recent_samples = RingBuffer(WINDOW_LENGTH)
       
       
       
            step=0
 
            while True:
            #for step in range(max_episode_length):
            
                step+=1
                
                #convert to uint8, prepare the state to be stored in frame, and buffer
                mem_proc_state=self.atari_proc.process_frame_for_memory(state)
                assert mem_proc_state is not None
                assert mem_proc_state.shape==(IMG_ROWS,IMG_COLS)
                
                #store the new observation in the buffer, it will be used for the selection of the next action
                recent_samples.append(mem_proc_state)
                
                
                if DEBUG_FRAMES:
                    for frame_id in range(0,len(recent_samples)):
                        sample_img=Image.fromarray(recent_samples[frame_id])
                        sample_img.save('sample_{0}_frame_{1}.jpg'.format(self.num_samples,frame_id))
            

                
                #merge samples in the buffer to create the network state
                net_proc_state=self.atari_proc.process_samples_for_network(recent_samples)
                
 
            
            
                assert net_proc_state.shape==(WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
                
                action = self.select_action(policy='training',state=[net_proc_state, self.input_dummymask])
                
                #apply the action to the environment, get reward and nextstate
                nextstate, reward, is_terminal, debug_info = env.step(action)
                training_total_reward +=reward
                
                assert nextstate is not None
                
                
                #append the new sample (state, action, reward, is next state terminal) in the memory
                self.memory.append(mem_proc_state, \
                                   action, \
                                   reward,\
                                   is_terminal,\
                                  True)


                
                state = nextstate
                
                #check if it's time to update the weights of the network
                if self.num_samples>0 and self.num_samples % self.train_freq == 0:
                    #print 'update weights'
                    self.update_policy()
                    self.num_updates += 1
                    
                    #check if it's time to evaluate the network
                    if self.num_updates>0 and self.num_updates % EVALUATION_FREQUENCY == 0:
                        #compute the reward, average episode length achieved in new episodes by the current agent
                        self.evaluate(eval_env,EVAL_NUM_EPISODES)
            
                #increase the total number (across all episodes) of interactions with the environment
                self.num_samples += 1
                print 'samples {0}'.format(self.num_samples)
                #check if the episode has finished
                
            

        
                if self.num_samples>0  and self.num_samples % SAVE_FREQUENCY==0:
                    self.save_model()
        
                #tensorboard training metrics
                QNAgent.save_scalar(self.num_updates,'total training reward',training_total_reward,self.writer)

                if is_terminal or (max_episode_length is not None and step == max_episode_length):

                    if DEBUG:
                        print 'Game terminates after {0} samples and {1} updates'.format(self.num_samples-self.num_burn_in, self.num_updates)
                    
                    break
            


            #clear the buffer for the current episode, avoid combining frames from different episodes
            del recent_samples


    #computes the q-values achieved by the current network on the random states of the burn-in phase
    def eval_avg_q(self):
        best_q=np.amax(self.calc_q_values([self.eval_states, self.eval_states_mask]))
        avg_q=np.mean(np.amax(self.calc_q_values([self.eval_states, self.eval_states_mask]), axis=1))
        return avg_q

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """

        if DEBUG:
            print "======================= evaluating source network ============================="
        
        total_reward = 0
        episode_length = 0
        for episode_idx in range(num_episodes):
        
            state=env.reset()
            
            #keep the frames of the state in a ring buffer
            recent_samples = RingBuffer(WINDOW_LENGTH)
            assert recent_samples is not None
        
            if DEBUG:
                print 'Evaluating episode {0}'.format(episode_idx)
            step=0
            
            while True:
                
                step+=1
                #convert to uint8, prepare the state to be stored in frame, and buffer
                mem_proc_state=self.atari_proc.process_frame_for_memory(state)
                assert mem_proc_state is not None
                assert mem_proc_state.shape==(IMG_ROWS,IMG_COLS)
                
                #store the new observation in the buffer, it will be used for the selection of the next action
                recent_samples.append(mem_proc_state)
                
                #merge samples in the buffer to create the network state
                net_proc_state=self.atari_proc.process_samples_for_network(recent_samples)
                assert net_proc_state.shape==(WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
                
                #select the next action
                action = self.select_action(policy='testing',state=[net_proc_state, self.input_dummymask])
                
                #compute the next state, the reward
                state, reward, is_terminal, debug_info = env.step(action)
                
                
                total_reward+=reward
                episode_length+=1
    
                
                #new episode should start
                if is_terminal or (max_episode_length is not None and step == max_episode_length):
                    break
        
            if DEBUG:
                print 'Episode {0} Reward {1}'.format(episode_idx, total_reward)
            #clear the buffer for the current episode
            del recent_samples

        
        QNAgent.save_scalar(self.num_updates,'testing total reward',total_reward,self.writer)
        QNAgent.save_scalar(self.num_updates,'testing episode length ',episode_length/num_episodes,self.writer)
    
    
    def update_policy(self):
        """Update your policy.
            
            Behavior may differ based on what stage of training your
            in. If you're in training mode then you should check if you
            should update your network parameters based on the current
            step and the value you set for train_freq.
            
            Inside, you'll want to sample a minibatch, calculate the
            target values, update your network, and then update your
            target values.
            
            You might want to return the loss and other metrics as an
            output. They can help you monitor how training is going.
            """

        raise NotImplementedError('This method should be overriden.')

    def save_model(self):
        raise NotImplementedError('This method should be overriden.')
    
    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.
            
            This is inspired by the compile method on the
            keras.models.Model class.
            
            This is a good place to create the target network, setup your
            loss function and any placeholders you might need.
            
            You should use the mean_huber_loss function as your
            loss_function. You can also experiment with MSE and other
            losses.
            
            The optimizer can be whatever class you want. We used the
            keras.optimizers.Optimizer class. Specifically the Adam
            optimizer.
            """
        
        raise NotImplementedError('This method should be overriden.')

class FTDQNAgent(QNAgent):
    
    """Class implementing fixed target DQN.
        
        This is a basic outline of the functions/parameters you will need
        in order to implement the DQNAgnet. This is just to get you
        started. You may need to tweak the parameters, add new ones, etc.
        
        Feel free to change the functions and funciton parameters that the
        class provides.
        
        We have provided docstrings to go along with our suggested API.
        
        Parameters
        ----------
        q_network: keras.models.Model
        Your Q-network model.
        preprocessor: deeprl_hw2.core.Preprocessor
        The preprocessor class. See the associated classes for more
        details.
        memory: deeprl_hw2.core.Memory
        Your replay memory.
        gamma: float
        Discount factor.
        target_update_freq: float
        Frequency to update the target network. You can either provide a
        number representing a soft target update (see utils.py) or a
        hard target update (see utils.py and Atari paper.)
        num_burn_in: int
        Before you begin updating the Q-network your replay memory has
        to be filled up with some number of samples. This number says
        how many.
        train_freq: int
        How often you actually update your Q-Network. Sometimes
        stability is improved if you collect a couple samples for your
        replay memory, for every Q-network update that you run.
        batch_size: int
        How many samples in each minibatch.
        """
    def __init__(self,
                 network_type,
                 num_actions,
                 preprocessors,
                 memory,
                 burnin_policy,
                 training_policy,
                 testing_policy,
                 gamma=GAMMA,
                 alpha=ALPHA,
                 target_update_freq=TARGET_QNET_RESET_INTERVAL,
                 num_burn_in=SAMPLES_BURN_IN,
                 train_freq=TRAINING_FREQUENCY,
                 eval_freq=EVALUATION_FREQUENCY,
                 batch_size=BATCH_SIZE):

        QNAgent.__init__(self,network_type,'FTDQN',num_actions,preprocessors,memory,burnin_policy,training_policy,testing_policy,gamma,alpha,target_update_freq,num_burn_in,train_freq,eval_freq,batch_size)

        if network_type=='LINEAR':
          #fixed-target network
	      self.qt_network  	= self.create_linear_model(window = WINDOW_LENGTH, \
							input_shape = (IMG_ROWS, IMG_COLS), \
							num_actions = self.num_actions)
    
	      self.q_network   	= self.create_linear_model(window = WINDOW_LENGTH, \
						    input_shape = (IMG_ROWS, IMG_COLS), \
						    num_actions = self.num_actions)
        
        elif network_type=='DEEP':
          #fixed-target network
	      self.qt_network  	= QNAgent.create_deep_model(window = WINDOW_LENGTH, \
						   input_shape = (IMG_ROWS, IMG_COLS), \
						   num_actions = self.num_actions)
                              
	      self.q_network   	= QNAgent.create_deep_model(window = WINDOW_LENGTH, \
						   input_shape = (IMG_ROWS, IMG_COLS), \
						   num_actions = self.num_actions)


        self.writer.add_graph(tf.get_default_graph())
            
    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.
        
            This is inspired by the compile method on the
            keras.models.Model class.
        
            This is a good place to create the target network, setup your
            loss function and any placeholders you might need.
        
            You should use the mean_huber_loss function as your
            loss_function. You can also experiment with MSE and other
            losses.
        
            The optimizer can be whatever class you want. We used the
            keras.optimizers.Optimizer class. Specifically the Adam
            optimizer.
        """
            
        if optimizer == 'Adam':
            opti = optimizers.Adam(lr=self.alpha)
            self.q_network.compile(loss=loss_func, optimizer = opti)

    def update_policy(self):
        """Update your policy.
            
            Behavior may differ based on what stage of training your
            in. If you're in training mode then you should check if you
            should update your network parameters based on the current
            step and the value you set for train_freq.
            
            Inside, you'll want to sample a minibatch, calculate the
            target values, update your network, and then update your
            target values.
            
            You might want to return the loss and other metrics as an
            output. They can help you monitor how training is going.
        """
                
        # sample the memory replay to get entries <state, action, reward (apply action at state), is_terminal (next state is terminal)>
        print 'Update network sample {0}'.format(self.num_samples)
        mem_samples = self.memory.sample(self.batch_size)
        assert len(mem_samples)==self.batch_size
        
        if DEBUG_FRAMES:
            for sample_id in range(0,self.batch_size):
                state=mem_samples[sample_id].state
                next_state=mem_samples[sample_id].next_state
                print 'Mem sample {0} action {1} reward {2}'.format(sample_id,mem_samples[sample_id].action,mem_samples[sample_id].reward)
                for fr_id in range(0,WINDOW_LENGTH):
                    next_state_img=Image.fromarray(next_state[fr_id])
                    next_state_img.save('mem_sample_{0}_next_state_{1}.jpg'.format(sample_id,fr_id))
                    state_img=Image.fromarray(state[fr_id])
                    state_img.save('mem_sample_{0}_state_{1}.jpg'.format(sample_id,fr_id))
    
        #the state-batch
        input_state_batch=np.zeros((self.batch_size, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))
        
        #the next-batch
        input_nextstate_batch=np.zeros((self.batch_size, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))
        
        #input mask needed to chose only one q-value
        input_mask_batch=np.zeros((self.batch_size,self.num_actions))
        
        #the q-value which corresponds to the applied action of the sample
        output_target_batch=np.zeros((self.batch_size,self.num_actions))
        
        
        for idx in range(self.batch_size):
            
            #create a 4d array with the states
            assert mem_samples[idx].state[0].shape==(IMG_ROWS,IMG_COLS)
            assert len(mem_samples[idx].state)==WINDOW_LENGTH
            input_state_batch[idx,:,:,:] = self.atari_proc.process_state_for_network(mem_samples[idx].state)
            
            #create a 4d array with the states
            assert mem_samples[idx].next_state[0].shape==(IMG_ROWS,IMG_COLS)
            assert len(mem_samples[idx].next_state)==WINDOW_LENGTH
            input_nextstate_batch[idx,:,:,:] = self.atari_proc.process_state_for_network(mem_samples[idx].next_state)
            
            #activate the output of the applied action
            input_mask_batch[idx, mem_samples[idx].action] = 1
        
        assert input_state_batch.shape==(self.batch_size,WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
        assert input_nextstate_batch.shape==(self.batch_size,WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
        
        #on the next state, chose the best predicted q-value on the fixed-target network
        target_q = self.qt_network.predict([input_nextstate_batch, self.input_dummymask_batch],batch_size=self.batch_size)
        
        
        assert target_q.shape==(self.batch_size,self.num_actions)
        best_target_q = np.amax(target_q, axis=1)
        
        
        assert best_target_q.shape==(self.batch_size,)
        

        
        #compute the target q-value r+gamma*max{a'}(Q(nextstat,a',qt)
        for ind in range(self.batch_size):

            output_target_batch[ind, mem_samples[ind].action] = mem_samples[ind].reward + self.gamma*best_target_q[ind]


        loss = self.q_network.train_on_batch(x=[input_state_batch, input_mask_batch], y=output_target_batch)
        QNAgent.save_scalar(self.num_updates, 'loss', loss, self.writer)
       


        #update the target network
        if self.num_updates % self.target_update_freq == 0:
            if DEBUG:
                print "======================= Sync target and source network ============================="
            tfrl.utils.get_hard_target_model_updates(self.qt_network, self.q_network)
                
        
    def save_model(self):
        self.q_network.save_weights('ftdqn_source_{0}.weight'.format(self.network_type))
        self.qt_network.save_weights('ftdqn_target_{0}.weight'.format(self.network_type))


class DoubleDQNAgent(QNAgent):
    
    """Class implementing Double QN.
        
        This is a basic outline of the functions/parameters you will need
        in order to implement the DQNAgnet. This is just to get you
        started. You may need to tweak the parameters, add new ones, etc.
        
        Feel free to change the functions and funciton parameters that the
        class provides.
        
        We have provided docstrings to go along with our suggested API.
        
        Parameters
        ----------
        q_network: keras.models.Model
        Your Q-network model.
        preprocessor: deeprl_hw2.core.Preprocessor
        The preprocessor class. See the associated classes for more
        details.
        memory: deeprl_hw2.core.Memory
        Your replay memory.
        gamma: float
        Discount factor.
        target_update_freq: float
        Frequency to update the target network. You can either provide a
        number representing a soft target update (see utils.py) or a
        hard target update (see utils.py and Atari paper.)
        num_burn_in: int
        Before you begin updating the Q-network your replay memory has
        to be filled up with some number of samples. This number says
        how many.
        train_freq: int
        How often you actually update your Q-Network. Sometimes
        stability is improved if you collect a couple samples for your
        replay memory, for every Q-network update that you run.
        batch_size: int
        How many samples in each minibatch.
        """
    def __init__(self,
                 network_type,
                 num_actions,
                 preprocessors,
                 memory,
                 burnin_policy,
                 training_policy,
                 testing_policy,
                 gamma=GAMMA,
                 alpha=ALPHA,
                 target_update_freq=TARGET_QNET_RESET_INTERVAL,
                 num_burn_in=SAMPLES_BURN_IN,
                 train_freq=TRAINING_FREQUENCY,
                 eval_freq=EVALUATION_FREQUENCY,
                 batch_size=BATCH_SIZE):
        
        QNAgent.__init__(self,network_type,'DoubleDqn',num_actions,preprocessors,memory,burnin_policy,training_policy,testing_policy,gamma,alpha,target_update_freq,num_burn_in,train_freq,eval_freq,batch_size)
        
        if network_type=='LINEAR':
            self.qt_network  	= self.create_linear_model(window = WINDOW_LENGTH, \
                                                       input_shape = (IMG_ROWS, IMG_COLS), \
                                                       num_actions = self.num_actions, \
                                                       )

            self.q_network   	= self.create_linear_model(window = WINDOW_LENGTH, \
                                                       input_shape = (IMG_ROWS, IMG_COLS), \
                                                       num_actions = self.num_actions
                                                       )
        elif network_type=='DEEP':

            self.qt_network  	= QNAgent.create_deep_model(window = WINDOW_LENGTH, \
                                                     input_shape = (IMG_ROWS, IMG_COLS), \
                                                     num_actions = self.num_actions
                                                     )

            self.q_network   	= QNAgent.create_deep_model(window = WINDOW_LENGTH, \
                                                     input_shape = (IMG_ROWS, IMG_COLS), \
                                                     num_actions = self.num_actions
                                                     )


    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.
            
            This is inspired by the compile method on the
            keras.models.Model class.
            
            This is a good place to create the target network, setup your
            loss function and any placeholders you might need.
            
            You should use the mean_huber_loss function as your
            loss_function. You can also experiment with MSE and other
            losses.
            
            The optimizer can be whatever class you want. We used the
            keras.optimizers.Optimizer class. Specifically the Adam
            optimizer.
            """
        
        if optimizer == 'Adam':
            opti = optimizers.Adam(lr=self.alpha)
            self.q_network.compile(loss=loss_func, optimizer = opti)

    def update_policy(self):
        """Update your policy.
            
            Behavior may differ based on what stage of training your
            in. If you're in training mode then you should check if you
            should update your network parameters based on the current
            step and the value you set for train_freq.
            
            Inside, you'll want to sample a minibatch, calculate the
            target values, update your network, and then update your
            target values.
            
            You might want to return the loss and other metrics as an
            output. They can help you monitor how training is going.
            """
            
        # sample the memory replay to get entries <state, action, reward (apply action at state), is_terminal (next state is terminal)>
        print 'Update network sample {0}'.format(self.num_samples)
        mem_samples = self.memory.sample(self.batch_size)
        assert len(mem_samples)==self.batch_size
        
        if DEBUG_FRAMES:
            for sample_id in range(0,self.batch_size):
                state=mem_samples[sample_id].state
                next_state=mem_samples[sample_id].next_state
                print 'Mem sample {0} action {1} reward {2}'.format(sample_id,mem_samples[sample_id].action,mem_samples[sample_id].reward)
                for fr_id in range(0,WINDOW_LENGTH):
                    next_state_img=Image.fromarray(next_state[fr_id])
                    next_state_img.save('mem_sample_{0}_next_state_{1}.jpg'.format(sample_id,fr_id))
                    state_img=Image.fromarray(state[fr_id])
                    state_img.save('mem_sample_{0}_state_{1}.jpg'.format(sample_id,fr_id))
        
        #the state-batch
        input_state_batch=np.zeros((self.batch_size, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))
        
        #the next-batch
        input_nextstate_batch=np.zeros((self.batch_size, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))
        
        #input mask needed to chose only one q-value
        input_mask_batch=np.zeros((self.batch_size,self.num_actions))
        
        #the q-value which corresponds to the applied action of the sample
        output_target_batch=np.zeros((self.batch_size,self.num_actions))
        
        
        for idx in range(self.batch_size):
        
            #create a 4d array with the states
            assert mem_samples[idx].state[0].shape==(IMG_ROWS,IMG_COLS)
            assert len(mem_samples[idx].state)==WINDOW_LENGTH
            input_state_batch[idx,:,:,:] = self.atari_proc.process_state_for_network(mem_samples[idx].state)
            
            #create a 4d array with the states
            assert mem_samples[idx].next_state[0].shape==(IMG_ROWS,IMG_COLS)
            assert len(mem_samples[idx].next_state)==WINDOW_LENGTH
            input_nextstate_batch[idx,:,:,:] = self.atari_proc.process_state_for_network(mem_samples[idx].next_state)
            
            #activate the output of the applied action
            input_mask_batch[idx, mem_samples[idx].action] = 1
        
        assert input_state_batch.shape==(self.batch_size,WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
        assert input_nextstate_batch.shape==(self.batch_size,WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
        
        if DEBUG_FRAMES:
            print input_state_batch
            print input_nextstate_batch

        #find the best action that can be applied on nextstate, given the q-values predicted by the network
        aux_q = self.q_network.predict([input_nextstate_batch, self.input_dummymask_batch],batch_size=1)

        
        assert aux_q.shape==(self.batch_size,self.num_actions)
        best_actions=np.argmax(aux_q,axis=1)
        assert best_actions.shape==(self.batch_size,)
  
        
        #keep the q-value of the target network, corresponding to the best actions
        target_q = self.qt_network.predict([input_nextstate_batch, self.input_dummymask_batch],batch_size=1)

        assert target_q.shape==(self.batch_size,self.num_actions)
        best_target_q = target_q[range(self.batch_size), best_actions]

        assert best_target_q.shape==(self.batch_size,)
    
        #target q-value for the state
        for ind in range(self.batch_size):

            output_target_batch[ind, mem_samples[ind].action] = mem_samples[ind].reward + self.gamma*best_target_q[ind]

        assert output_target_batch.shape==(self.batch_size,self.num_actions)

        
        loss = self.q_network.train_on_batch(x=[input_state_batch, input_mask_batch], y=output_target_batch)
        QNAgent.save_scalar(self.num_updates, 'loss', loss, self.writer)
     
    
    
        #update the target network
        if self.num_updates % self.target_update_freq == 0:
            if DEBUG:
                print "======================= Sync target and source network ============================="
            tfrl.utils.get_hard_target_model_updates(self.qt_network, self.q_network)


    def save_model(self):
        self.q_network.save_weights('doubledqn_source_{0}.weight'.format(self.network_type))
        self.qt_network.save_weights('doubledqn_target_{0}.weight'.format(self.network_type))
    


class DQNAgent(QNAgent):
    
    """Class implementing the classic (not fixed-target) DQN.
        
        This is a basic outline of the functions/parameters you will need
        in order to implement the DQNAgnet. This is just to get you
        started. You may need to tweak the parameters, add new ones, etc.
        
        Feel free to change the functions and funciton parameters that the
        class provides.
        
        We have provided docstrings to go along with our suggested API.
        
        Parameters
        ----------
        q_network: keras.models.Model
        Your Q-network model.
        preprocessor: deeprl_hw2.core.Preprocessor
        The preprocessor class. See the associated classes for more
        details.
        memory: deeprl_hw2.core.Memory
        Your replay memory.
        gamma: float
        Discount factor.
        target_update_freq: float
        Frequency to update the target network. You can either provide a
        number representing a soft target update (see utils.py) or a
        hard target update (see utils.py and Atari paper.)
        num_burn_in: int
        Before you begin updating the Q-network your replay memory has
        to be filled up with some number of samples. This number says
        how many.
        train_freq: int
        How often you actually update your Q-Network. Sometimes
        stability is improved if you collect a couple samples for your
        replay memory, for every Q-network update that you run.
        batch_size: int
        How many samples in each minibatch.
        """
    def __init__(self,
             network_type,
             num_actions,
             preprocessors,
             memory,
             burnin_policy,
             training_policy,
             testing_policy,
             gamma=GAMMA,
             alpha=ALPHA,
             target_update_freq=TARGET_QNET_RESET_INTERVAL,
             num_burn_in=SAMPLES_BURN_IN,
             train_freq=TRAINING_FREQUENCY,
             eval_freq=EVALUATION_FREQUENCY,
             batch_size=BATCH_SIZE):
    
            QNAgent.__init__(self,network_type,'DQN',num_actions,preprocessors,memory,burnin_policy,training_policy,testing_policy,gamma,alpha,target_update_freq,num_burn_in,train_freq,eval_freq,batch_size)
        
        
            if network_type=='LINEAR':
                self.q_network = QNAgent.create_linear_model(window = WINDOW_LENGTH, \
                                                          input_shape = (IMG_ROWS, IMG_COLS), \
                                                          num_actions = self.num_actions
                                                          )
            elif network_type=='DEEP':
                self.q_network = QNAgent.create_deep_model(window = WINDOW_LENGTH, \
                                                        input_shape = (IMG_ROWS, IMG_COLS), \
                                                        num_actions = self.num_actions
                                                        )

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.
            
            This is inspired by the compile method on the
            keras.models.Model class.
            
            This is a good place to create the target network, setup your
            loss function and any placeholders you might need.
            
            You should use the mean_huber_loss function as your
            loss_function. You can also experiment with MSE and other
            losses.
            
            The optimizer can be whatever class you want. We used the
            keras.optimizers.Optimizer class. Specifically the Adam
            optimizer.
            """
            
        if optimizer == 'Adam':
        	opti = optimizers.Adam(lr=self.alpha)
        	self.q_network.compile(loss=loss_func, optimizer = opti)
        
    def update_policy(self):
        """Update your policy.
            
            Behavior may differ based on what stage of training your
            in. If you're in training mode then you should check if you
            should update your network parameters based on the current
            step and the value you set for train_freq.
            
            Inside, you'll want to sample a minibatch, calculate the
            target values, update your network, and then update your
            target values.
            
            You might want to return the loss and other metrics as an
            output. They can help you monitor how training is going.
            """

        # sample the memory replay to get entries <state, action, reward (apply action at state), is_terminal (next state is terminal)>
        print 'Update network sample {0}'.format(self.num_samples)
        mem_samples = self.memory.sample(self.batch_size)
        assert len(mem_samples)==self.batch_size
    
        #the state-batch
        input_state_batch=np.zeros((self.batch_size, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))
    
        #the next-batch
        input_nextstate_batch=np.zeros((self.batch_size, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))
    
        #input mask needed to chose only one q-value
        input_mask_batch=np.zeros((self.batch_size,self.num_actions))
    
        #the q-value which corresponds to the applied action of the sample
        output_target_batch=np.zeros((self.batch_size,self.num_actions))
    
    
        for idx in range(self.batch_size):
        
            #create a 4d array with the states
            assert mem_samples[idx].state[0].shape==(IMG_ROWS,IMG_COLS)
            assert len(mem_samples[idx].state)==WINDOW_LENGTH
            input_state_batch[idx,:,:,:] = self.atari_proc.process_state_for_network(mem_samples[idx].state)
        
            #create a 4d array with the states
            assert mem_samples[idx].next_state[0].shape==(IMG_ROWS,IMG_COLS)
            assert len(mem_samples[idx].next_state)==WINDOW_LENGTH
            input_nextstate_batch[idx,:,:,:] = self.atari_proc.process_state_for_network(mem_samples[idx].next_state)
        
            #activate the output of the applied action
            input_mask_batch[idx, mem_samples[idx].action] = 1

        assert input_state_batch.shape==(self.batch_size,WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
        assert input_nextstate_batch.shape==(self.batch_size,WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
    
        #on the next state, chose the best predicted q-value on the network
        target_q = self.q_network.predict([input_nextstate_batch, self.input_dummymask_batch],batch_size=self.batch_size)
        

        assert target_q.shape==(self.batch_size,self.num_actions)
        best_target_q = np.amax(target_q, axis=1)
        

    
        assert best_target_q.shape==(self.batch_size,)
    

    
        #compute the target q-value r+gamma*max{a'}(Q(nextstat,a',qt)
        for ind in range(self.batch_size):
            output_target_batch[ind, mem_samples[ind].action] = mem_samples[ind].reward + self.gamma*best_target_q[ind]


        loss = self.q_network.train_on_batch(x=[input_state_batch, input_mask_batch], y=output_target_batch)
        QNAgent.save_scalar(self.num_updates, 'loss', loss, self.writer)





    def save_model(self):
        self.q_network.save_weights('dqn_qn_source_{0}.weight'.format(self.network_type))



class DuelingDQNAgent(QNAgent):
    
    """Class implementing Dueling QN.
        
        This is a basic outline of the functions/parameters you will need
        in order to implement the DQNAgnet. This is just to get you
        started. You may need to tweak the parameters, add new ones, etc.
        
        Feel free to change the functions and funciton parameters that the
        class provides.
        
        We have provided docstrings to go along with our suggested API.
        
        Parameters
        ----------
        q_network: keras.models.Model
        Your Q-network model.
        preprocessor: deeprl_hw2.core.Preprocessor
        The preprocessor class. See the associated classes for more
        details.
        memory: deeprl_hw2.core.Memory
        Your replay memory.
        gamma: float
        Discount factor.
        target_update_freq: float
        Frequency to update the target network. You can either provide a
        number representing a soft target update (see utils.py) or a
        hard target update (see utils.py and Atari paper.)
        num_burn_in: int
        Before you begin updating the Q-network your replay memory has
        to be filled up with some number of samples. This number says
        how many.
        train_freq: int
        How often you actually update your Q-Network. Sometimes
        stability is improved if you collect a couple samples for your
        replay memory, for every Q-network update that you run.
        batch_size: int
        How many samples in each minibatch.
        """
    def __init__(self,
             network_type,
             num_actions,
             preprocessors,
             memory,
             burnin_policy,
             training_policy,
             testing_policy,
             gamma=GAMMA,
             alpha=ALPHA,
             target_update_freq=TARGET_QNET_RESET_INTERVAL,
             num_burn_in=SAMPLES_BURN_IN,
             train_freq=TRAINING_FREQUENCY,
             eval_freq=EVALUATION_FREQUENCY,
             batch_size=BATCH_SIZE):

            QNAgent.__init__(self,network_type,num_actions,preprocessors,memory,burnin_policy,training_policy,testing_policy,gamma,alpha,target_update_freq,num_burn_in,train_freq,eval_freq,batch_size)


            self.qt_network = self.create_dueling_model(window = WINDOW_LENGTH, \
                        input_shape = (IMG_ROWS, IMG_COLS), \
                        num_actions = self.num_actions)

            self.q_network = self.create_dueling_model(window = WINDOW_LENGTH, \
                        input_shape = (IMG_ROWS, IMG_COLS), \
                        num_actions = self.num_actions)



    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.
        
            This is inspired by the compile method on the
            keras.models.Model class.
        
            This is a good place to create the target network, setup your
            loss function and any placeholders you might need.
        
            You should use the mean_huber_loss function as your
            loss_function. You can also experiment with MSE and other
            losses.
        
            The optimizer can be whatever class you want. We used the
            keras.optimizers.Optimizer class. Specifically the Adam
            optimizer.
        """
            
        if optimizer == 'Adam':
            opti = optimizers.Adam(lr=self.alpha)
            self.q_network.compile(loss=loss_func, optimizer = opti)


    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.
            
            This is inspired by the compile method on the
            keras.models.Model class.
            
            This is a good place to create the target network, setup your
            loss function and any placeholders you might need.
            
            You should use the mean_huber_loss function as your
            loss_function. You can also experiment with MSE and other
            losses.
            
            The optimizer can be whatever class you want. We used the
            keras.optimizers.Optimizer class. Specifically the Adam
            optimizer.
            """
            
        if optimizer == 'Adam':
            opti = optimizers.Adam(lr=self.alpha)
            self.q_network.compile(loss=loss_func, optimizer = opti)
    
    def update_policy(self):
        """Update your policy.
            
            Behavior may differ based on what stage of training your
            in. If you're in training mode then you should check if you
            should update your network parameters based on the current
            step and the value you set for train_freq.
            
            Inside, you'll want to sample a minibatch, calculate the
            target values, update your network, and then update your
            target values.
            
            You might want to return the loss and other metrics as an
            output. They can help you monitor how training is going.
            """
        print 'Update network sample {0}'.format(self.num_samples)
        # sample the memory replay to get entries <state, action, reward (apply action at state), is_terminal (next state is terminal)>
        mem_samples = self.memory.sample(self.batch_size)
        assert len(mem_samples)==self.batch_size

        #the state-batch
        input_state_batch=np.zeros((self.batch_size, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))

        #the next-batch
        input_nextstate_batch=np.zeros((self.batch_size, WINDOW_LENGTH, IMG_ROWS, IMG_COLS))

        #input mask needed to chose only one q-value
        input_mask_batch=np.zeros((self.batch_size,self.num_actions))

        #the q-value which corresponds to the applied action of the sample
        output_target_batch=np.zeros((self.batch_size,self.num_actions))
    
    
        for idx in range(self.batch_size):
            
            #create a 4d array with the states
            assert mem_samples[idx].state[0].shape==(IMG_ROWS,IMG_COLS)
            assert len(mem_samples[idx].state)==WINDOW_LENGTH
            input_state_batch[idx,:,:,:] = self.atari_proc.process_state_for_network(mem_samples[idx].state)
            
            #create a 4d array with the states
            assert mem_samples[idx].next_state[0].shape==(IMG_ROWS,IMG_COLS)
            assert len(mem_samples[idx].next_state)==WINDOW_LENGTH
            input_nextstate_batch[idx,:,:,:] = self.atari_proc.process_state_for_network(mem_samples[idx].next_state)
            
            #activate the output of the applied action
            input_mask_batch[idx, mem_samples[idx].action] = 1

        assert input_state_batch.shape==(self.batch_size,WINDOW_LENGTH,IMG_ROWS,IMG_COLS)
        assert input_nextstate_batch.shape==(self.batch_size,WINDOW_LENGTH,IMG_ROWS,IMG_COLS)

        #on the next state, chose the best predicted q-value on the fixed-target network
        target_q = self.qt_network.predict([input_nextstate_batch, self.input_dummymask_batch],batch_size=self.batch_size)
        assert target_q.shape==(self.batch_size,self.num_actions)
        best_target_q = np.amax(target_q, axis=1)

        assert best_target_q.shape==(self.batch_size,)

        #compute the target q-value r+gamma*max{a'}(Q(nextstat,a',qt)
        for ind in range(self.batch_size):
            output_target_batch[ind, mem_samples[ind].action] = mem_samples[ind].reward + self.gamma*best_target_q[ind]

        loss = self.q_network.train_on_batch(x=[input_state_batch, input_mask_batch], y=output_target_batch)
      
      


        #update the target network
        if self.num_updates % self.target_update_freq == 0:
            print "======================= Sync target and source network ============================="
            tfrl.utils.get_hard_target_model_updates(self.qt_network, self.q_network)


    def save_model(self):
        self.q_network.save_weights('dueldqn_source_{0}.weight'.format(self.network_type))
        self.qt_network.save_weights('dueldqn_target_{0}.weight'.format(self.network_type))

