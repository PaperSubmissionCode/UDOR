import os, sys
import time
import re
import numpy as np
import tensorflow as tf

import lib.models as lib
from lib.models import params_with_name
from lib.models.save_images import save_images
from lib.models.distributions import Bernoulli, Gaussian, Product
from lib.models.nets_32x32_small import NetsRetreiver, NetsRetreiverWithClassifier

TINY = 1e-8
SEED = 123
ch=1
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter

class AE(object):
    def __init__(self, session, arch,lr,alpha,beta,latent_dim,latent_num,class_net_unit_num,output_dim, batch_size, image_shape, exp_name, dirs,
        vis_reconst):
        """
        :type output_dist: Distribution
        :type z_dist: Gaussian
        """
        self.session = session
        self.arch = arch
        self.lr=lr
        self.alpha=alpha
        self.beta=beta
        self.latent_dim=latent_dim
        self.latent_num=latent_num
        self.class_net_unit_num=class_net_unit_num
        self.output_dim=output_dim
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.exp_name = exp_name
        self.dirs = dirs
        self.vis_reconst = vis_reconst
        
        self.__build_graph()

    def __build_graph(self):
        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        self.is_training = tf.placeholder(tf.bool)
        self.x1 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.x_gan = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.mask= tf.placeholder(tf.float32, shape=[None] + list([self.latent_dim]))
        self.mask_zero=tf.placeholder(tf.float32, shape=[None] + list([self.latent_dim]))
        self.img_black=tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        # classification gt
        self.vec_one=tf.placeholder(tf.float32, shape=[None] + list([1])) # which used in the classification loss
        self.class_gt0=tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+1]))
        self.class_gt1=tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+1]))
        self.class_gt2=tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+1]))
        self.class_gt3=tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+1]))
        self.class_gt4=tf.placeholder(tf.float32, shape=[None] + list([self.latent_num+1]))
        
        # Normalize + reshape 'real' input data
        norm_x1 = 2*(tf.cast(self.x1, tf.float32)-.5)
        norm_img_black=2*(tf.cast(self.img_black, tf.float32)-.5)
        norm_x_gan=2*(tf.cast(self.x_gan, tf.float32)-.5)
        # Set Encoder and Decoder archs
        self.Encoder, self.Decoder,self.Classifier,self.gan_discriminator = NetsRetreiverWithClassifier(self.arch) 
    
        # Encode
        self.z1 = self.__Enc(norm_x1)
        # original stage
        # Decode
        self.x_out1 = self.__Dec(self.z1)
        # random set 0
        self.r1=tf.multiply(self.z1,self.mask)
        self.x_out_r0 = self.__Dec(self.r1)
        r11 = self.__Enc(self.x_out_r0)

        # set representation all 000
        self.r_zero=tf.multiply(self.z1,self.mask_zero)
        self.img_out_zero=self.__Dec(self.r_zero)
        # split latent representation into 2 part and  classification
        r_part1,r_part2,r_part3,r_part4=tf.split(self.r1,4,axis=1)
        r_all0,r_all0,r_all0,r_all0=tf.split(self.r_zero,4,axis=1)

        c_p0=self.__Classifier(r_all0)
        c_p1=self.__Classifier(r_part1)
        c_p2=self.__Classifier(r_part2)
        c_p3=self.__Classifier(r_part3)
        c_p4=self.__Classifier(r_part4)
        #======================================================
        # swap part (used in the test time for visualization)
        self.x2 = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
        self.mask2= tf.placeholder(tf.float32, shape=[None] + list([self.latent_dim]))
        norm_x2 = 2*(tf.cast(self.x2, tf.float32)-.5)
        self.z2 = self.__Enc(norm_x2)
        self.r2=tf.multiply(self.z2,self.mask2)
        self.r12=tf.add(self.r1,self.r2)
        self.x_swap=self.__Dec(self.r12)
        #=====================================================
        # GAN Discriminator
        self.fake_data=self.x_out_r0
        self.real_data=tf.reshape(norm_x_gan,[-1, self.image_shape[0]*self.image_shape[1]*self.image_shape[2]])
        #self.real_data=norm_x1
        disc_real = self.__GAN_discriminator(self.real_data)  # input real image data 
        disc_fake = self.__GAN_discriminator(self.fake_data)  # input fake image data
    
        # Loss and optimizer
        self.__prep_loss_optimizer(norm_x1,norm_img_black,r11,c_p0,c_p1,c_p2,c_p3,c_p4,disc_real,disc_fake)   

    def __Enc(self, x):
        #resnet_encoder(name, inputs, n_channels, latent_dim, is_training, mode=None, nonlinearity=tf.nn.relu):
        z= self.Encoder('Encoder', x, self.image_shape[0], self.latent_dim,self.is_training)
        return z
    
    def __Dec(self, z):
        x_out_logit = self.Decoder('Decoder', z, self.image_shape[0], self.is_training)
        x_out = tf.tanh(x_out_logit)
        return x_out
    
    def __Classifier(self,z):
        x_out= self.Classifier('Classifier', z, self.class_net_unit_num,self.latent_num+1, self.is_training)
        x_out = tf.nn.softmax(x_out)
        return x_out
    
    def __GAN_discriminator(self,x):
        dis_out=self.gan_discriminator('Discriminator',x,self.image_shape[0],self.is_training)
        return dis_out

    
    def __prep_loss_optimizer(self, norm_x1,norm_img_black,r11,c_p0,c_p1,c_p2,c_p3,c_p4,disc_real,disc_fake):
 
        norm_x1= tf.reshape(norm_x1, [-1, self.output_dim])
        norm_img_black= tf.reshape(norm_img_black, [-1, self.output_dim])
        #[Loss1]img reconstruction loss
        reconstr_img_loss =  tf.reduce_sum(tf.square(norm_x1 -self.x_out1), axis=1)   
        #[Loss2] rest 0 latent representation reconstruction loss
        reconstr__rep_loss =  tf.reduce_sum(tf.square(self.r1 -r11), axis=1) 
        #[Loss3]with representation all 0  image loss
        reconstr_img_zero_loss =  tf.reduce_sum(tf.square(norm_img_black -self.img_out_zero), axis=1) 
        #[loss4] classification loss
        temp_1=self.vec_one-tf.reduce_sum((self.class_gt0-self.class_gt0*c_p1),1)*tf.reduce_sum((self.class_gt1-self.class_gt1*c_p1),1)
        self.class1_loss=-tf.reduce_mean(tf.log(temp_1))
        temp_2=self.vec_one-tf.reduce_sum((self.class_gt0-self.class_gt0*c_p2),1)*tf.reduce_sum((self.class_gt2-self.class_gt2*c_p2),1)
        self.class2_loss=-tf.reduce_mean(tf.log(temp_2))
        temp_3=self.vec_one-tf.reduce_sum((self.class_gt0-self.class_gt0*c_p3),1)*tf.reduce_sum((self.class_gt3-self.class_gt3*c_p3),1)
        self.class3_loss=-tf.reduce_mean(tf.log(temp_3))
        temp_4=self.vec_one-tf.reduce_sum((self.class_gt0-self.class_gt0*c_p4),1)*tf.reduce_sum((self.class_gt4-self.class_gt4*c_p4),1)
        self.class4_loss=-tf.reduce_mean(tf.log(temp_4))
        #zero input class 1
        self.class0_loss=-tf.reduce_mean(self.class_gt0*tf.log(c_p0))*4
        

        # average over batch
        self.rec_loss=1.0*tf.reduce_mean(reconstr_img_loss)
        self.reset0_loss =1.0* tf.reduce_mean(reconstr__rep_loss)
        self.rec_zero_loss=10.0*tf.reduce_mean(reconstr_img_zero_loss)
        self.class_loss=1000.0*(self.class1_loss+self.class2_loss+self.class3_loss+self.class4_loss+self.class0_loss)

        self.loss=self.rec_loss+self.reset0_loss+self.rec_zero_loss+self.class_loss
        lr=self.lr
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0., beta2=0.9).minimize(self.loss) 
    
        print('Learning rate=')
        print(lr)
        # ==============GAN LOSS=============================
        self.gen_cost = -tf.reduce_mean(disc_fake)
        self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        alpha = tf.random_uniform(
        shape=[self.batch_size,1], 
        minval=0.,
        maxval=1.
        )

        differences = self.fake_data - self.real_data
        interpolates = self.real_data + (alpha*differences)
        gradients = tf.gradients(self.__GAN_discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        LAMBDA = 10 # Gradient penalty lambda hyperparameter
        self.disc_cost += LAMBDA*gradient_penalty

        self.gen_cost=self.gen_cost
        self.disc_cost=self.disc_cost
        gen_params = lib.params_with_name('Decoder')# Generator is Decoder
        disc_params = lib.params_with_name('Discriminator')
        self.gen_train_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(self.gen_cost, var_list=gen_params)
        self.disc_train_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(self.disc_cost, var_list=disc_params)


    def load(self):
        self.saver = tf.train.Saver(max_to_keep=9999999)
        ckpt = tf.train.get_checkpoint_state(self.dirs['ckpt'])
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = ckpt.model_checkpoint_path
            self.saver.restore(self.session, ckpt_name)
            print("Checkpoint restored: {0}".format(ckpt_name))
            prev_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print('prev_step=')
            print(prev_step)
        
        else:
            print("Failed to find checkpoint.")
            prev_step = 0
        sys.stdout.flush()
        return prev_step + 1

    def load_fixedNum(self,inter_num):
        #self.saver = tf.train.Saver()
        self.saver = tf.train.Saver(max_to_keep=99999999)
        ckpt = tf.train.get_checkpoint_state(self.dirs['ckpt'])
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = ckpt.model_checkpoint_path
            ckpt_name_prefix=ckpt_name.split('-')[0]
            ckpt_name_new=ckpt_name_prefix+'-'+str(inter_num)
            self.saver.restore(self.session, ckpt_name_new)
            print("Checkpoint restored: {0}".format(ckpt_name_new))
            prev_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name_new)).group(0))
            print('prev_step=')
            print(prev_step)
        else:
            print("Failed to find checkpoint.")
            prev_step = 0
        sys.stdout.flush()
        return prev_step + 1 
     
    def train(self, n_iters, n_iters_per_epoch, stats_iters, ckpt_interval):
        # for save loss  
        logArray=np.zeros((1000,4))
        count=0      
        self.session.run(tf.global_variables_initializer())
        
        # Fixed GT samples - save
        fixed_x1, fixed_mk1 , _ = next(self.train_iter1)
        print("fixed_mk1=")
        print(fixed_mk1[0:4])
        # replace mask
        unitLength=1 #(need to changed when has larger unitLength)
        
        # get classification gt and label
        class_gt0,class_gt1,class_gt2,class_gt3,class_gt4,vector_one=self.generateClassificationLabelandVecOne(self.batch_size)
        # generate zero representation and black image
        img_zero,fixed_zero_mk=self.generateMaskZero(self.batch_size,unitLength)
        #
        fixed_x1= self.session.run(tf.constant(fixed_x1))
        save_images(fixed_x1, os.path.join(self.dirs['samples'], 'samples_1_groundtruth.png'))
        #
        fixed_img_zero = self.session.run(tf.constant(img_zero*0.0))
        save_images(fixed_img_zero, os.path.join(self.dirs['samples'], 'samples_black_groundtruth.png'))
 
        start_iter = self.load()
        running_cost = 0.
        GAN_runing_gen_cost=0.
        GAN_runing_dis_cost=0.
        
        _gan_data=fixed_x1
        for iteration in range(start_iter, n_iters):
            start_time = time.time()

            
            _data1, _mask1, _ = next(self.train_iter1)

            _, cost = self.session.run((self.optimizer, self.loss),feed_dict={self.x1: _data1,self.mask: _mask1,self.vec_one:vector_one,self.img_black:img_zero,self.mask_zero:fixed_zero_mk,self.class_gt0:class_gt0,self.class_gt1:class_gt1,self.class_gt2:class_gt2,self.class_gt3:class_gt3,self.class_gt4:class_gt4,self.is_training:True})
            running_cost += cost
            #==generator training
            _,_disc_gene_cost=self.session.run((self.gen_train_optimizer,self.gen_cost),feed_dict={self.x1: _data1,self.x_gan:_gan_data,self.mask: _mask1,self.vec_one:vector_one,self.img_black:img_zero,self.mask_zero:fixed_zero_mk,self.class_gt0:class_gt0,self.class_gt1:class_gt1,self.class_gt2:class_gt2,self.class_gt3:class_gt3,self.class_gt4:class_gt4,self.is_training:True})
            GAN_runing_gen_cost+=_disc_gene_cost
            #==Discriminator training
            for i in range(CRITIC_ITERS):
                _,_disc_disc_cost = self.session.run((self.disc_train_optimizer,self.disc_cost),feed_dict={self.x1: _data1,self.x_gan: _gan_data,self.mask: _mask1,self.vec_one:vector_one,self.img_black:img_zero,self.mask_zero:fixed_zero_mk,self.class_gt0:class_gt0,self.class_gt1:class_gt1,self.class_gt2:class_gt2,self.class_gt3:class_gt3,self.class_gt4:class_gt4,self.is_training:True})
                GAN_runing_dis_cost+=_disc_disc_cost
            GAN_runing_dis_cost=GAN_runing_dis_cost/CRITIC_ITERS

            if iteration % n_iters_per_epoch == 1:
                print("Epoch: {0}".format(iteration // n_iters_per_epoch))
            
            # Print avg stats and dev set stats
            if (iteration < start_iter + 4) or iteration % stats_iters == 0:
                t = time.time()
                dev_data1, dev_mask1, _= next(self.dev_iter1)
                
                dev_cost,dev_rec_loss,dev_reset0_loss,rec_zero_loss,class_loss,gen_cost,disc_cost= self.session.run([self.loss,self.rec_loss,self.reset0_loss,self.rec_zero_loss,self.class_loss,self.gen_cost,self.disc_cost],feed_dict={self.x1: dev_data1,self.x_gan: _gan_data, self.mask: dev_mask1, self.vec_one:vector_one,self.img_black:img_zero,self.mask_zero:fixed_zero_mk,self.class_gt0:class_gt0,self.class_gt1:class_gt1,self.class_gt2:class_gt2,self.class_gt3:class_gt3,self.class_gt4:class_gt4,self.is_training:False})
                
                n_samples = 1. if (iteration < start_iter + 4) else float(stats_iters)
                avg_cost = running_cost / n_samples
                avg_GAN_runing_gen_cost=GAN_runing_gen_cost/n_samples
                avg_GAN_runing_dis_cost=GAN_runing_dis_cost/n_samples
                running_cost = 0.
                GAN_runing_gen_cost=0.
                GAN_runing_dis_cost=0.


                print("Iteration:{0} \t| Train cost:{1:.1f} \t| Dev cost: {2:.1f}(reconstr_loss:{3:.1f},reset0_loss:{4:.1f},rec_zero_loss:{5:.1f},class_loss:{6:.1f})".format(iteration, avg_cost, dev_cost,dev_rec_loss,dev_reset0_loss,rec_zero_loss,class_loss))
                print("Iteration:{0} \t| Train gen_cost:{1:.1f}  disc_cost:{2:.1f}\t| Dev gen_cost:{3:.1f}  disc_cost:{4:.1f}".format(iteration, avg_GAN_runing_gen_cost,avg_GAN_runing_dis_cost,gen_cost,disc_cost))
                
                # for save loss
                logArray[count,0]=iteration // n_iters_per_epoch
                logArray[count,1]=iteration
                logArray[count,2]=avg_cost
                logArray[count,3]=dev_cost
                count=count+1 
                if self.vis_reconst:
                    self.visualise_reconstruction(fixed_x1,fixed_mk1,iteration)
                      
                if np.any(np.isnan(avg_cost)):
                    raise ValueError("NaN detected!")            
            # save checkpoint
            if (iteration > start_iter) and iteration % (ckpt_interval) == 0:
                self.saver.save(self.session, os.path.join(self.dirs['ckpt'], self.exp_name), global_step=iteration)  
            _gan_data=_data1
        # for save loss
        np.save('logArray.npy',logArray) 

    def reconstruct(self, X1, mk1, is_training=False):
        """ Reconstruct data. """
        return self.session.run([self.x_out1,self.x_out_r0 ], 
                                feed_dict={self.x1: X1,self.mask: mk1, self.is_training: is_training})
    

    def visualise_reconstruction(self, X1,mk1,iteration):
        X_r1,X_r0= self.reconstruct(X1,mk1)
        #print(X_r0[3])
        X_r1 = ((X_r1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r0 = ((X_r0+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1, os.path.join(self.dirs['samples'], str(iteration)+'samples_reconstructed.png'))
        save_images(X_r0, os.path.join(self.dirs['samples'], str(iteration)+'reset0_reconstructed.png'))
        


    def encodeImg(self,pathForSave,X1, mk1,k, is_training=False): 
        
        X_r1,X_r0=self.session.run([self.x_out1,self.x_out_r0],feed_dict={self.x1: X1,self.mask: mk1, self.is_training: is_training})
        X_r1 = ((X_r1+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_r0 = ((X_r0+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_r1, os.path.join(pathForSave, 'iter'+str(k)+'_samples_reconstructed.png'))
        save_images(X_r0, os.path.join(pathForSave, 'iter'+str(k)+'_reset0_reconstructed.png'))
        
    def encode(self, X, is_training=False):
        """Encode data, i.e. map it into latent space."""
        code = self.session.run(self.z1, feed_dict={self.x1: X, self.is_training: is_training})
        return code

    def getCodesAndImgs(self, pathForSave, X1, mk1, k, is_training=False):
        z1, X_r0 = self.session.run([self.z1, self.x_out_r0],
                                    feed_dict={self.x1: X1, self.mask: mk1, self.is_training: is_training})
        ImageNorm0_1 = ((X_r0 + 1.) * (1.00 / 2)).astype('double').reshape(
            [-1, self.image_shape[1], self.image_shape[2], self.image_shape[0]])
        # for visual the first result to valide it effectiveness
        if k == 1:
            X_save = ((X_r0 + 1.) * (255.99 / 2)).astype('int32').reshape([-1] + self.image_shape)
            save_images(X_save, os.path.join(pathForSave, 'iter' + str(k) + '_samples_reconstructed.png'))
        return z1, ImageNorm0_1

    def getVisualImgs(self,pathForSave,X1, mk1,X2, mk2,k, is_training=False):
        X_r0,X_swap=self.session.run([self.x_out_r0,self.x_swap],feed_dict={self.x1: X1,self.mask: mk1,self.x2: X2,self.mask2: mk2,self.is_training: is_training})

        X_orig1_save = (X1*255.99).astype('int32').reshape([-1] + self.image_shape)
        X_orig2_save  = (X2*255.99).astype('int32').reshape([-1] + self.image_shape)
        X_reset0_save = ((X_r0+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        X_Swap_save = ((X_swap+1.)*(255.99/2)).astype('int32').reshape([-1] + self.image_shape)
        save_images(X_orig1_save, os.path.join(pathForSave, 'iter'+str(k)+'_orig_img.png'))
        save_images(X_orig2_save, os.path.join(pathForSave, 'iter'+str(k)+'_orig_swap.png'))
        save_images(X_reset0_save, os.path.join(pathForSave, 'iter'+str(k)+'_reset0_img.png'))
        save_images(X_Swap_save, os.path.join(pathForSave, 'iter'+str(k)+'_swap_img.png'))
        

    def generateMaskZero(self,batch_size,unitLength):
        #==============get mask==============
        maskArray=np.empty((batch_size,unitLength*4))
        mask=np.zeros((unitLength*4))
        # reset value 0~64
        for i in range(0,batch_size):
            maskArray[i]=mask
        w=32
        h=32
        imgArray= np.zeros((batch_size,ch,w,h))*0.0

        return imgArray,maskArray


    def generateClassificationLabelandVecOne(self,batch_size):
        #==============get mask==============
        class_num=5
        class_gt0=np.zeros((batch_size,class_num))
        class_gt1=np.zeros((batch_size,class_num))
        class_gt2=np.zeros((batch_size,class_num))
        class_gt3=np.zeros((batch_size,class_num))
        class_gt4=np.zeros((batch_size,class_num))

        for i in range(batch_size):
            class_gt0[i,0]=1
            class_gt1[i,1]=1
            class_gt2[i,2]=1
            class_gt3[i,3]=1
            class_gt4[i,4]=1
        vector_one=np.ones((batch_size,1))
        return class_gt0,class_gt1,class_gt2,class_gt3,class_gt4,vector_one