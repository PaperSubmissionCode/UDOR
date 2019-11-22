import os
import errno
import tensorflow as tf
from operator import mul
from functools import reduce
import sys

# ========= notes: also can be changed to your [absolute_local_path] ==========
sys.path.append('../')

from model import AE
from lib.models.data_managers_diae import TeapotsDataManager
from lib.utils import init_directories, create_directories

flags = tf.app.flags
flags.DEFINE_integer("epochs", 30, "Number of epochs to train [25]")
flags.DEFINE_integer("stats_interval", 1, "Print/log stats every [stats_interval] epochs. [1.0]")
flags.DEFINE_integer("ckpt_interval", 1, "Save checkpoint every [ckpt_interval] epochs. [1]")
flags.DEFINE_integer("latent_dim", 9 * 3, "Number of latent variables [1*3]")
flags.DEFINE_integer("latent_num", 3, "Number of latent part [3]")
flags.DEFINE_integer("class_net_unit_num", 9, "Number of neuro cell [9]")
flags.DEFINE_integer("batch_size", 64, "The size of training batches [64]")
flags.DEFINE_string("image_shape", "(3,64,64)", "Shape of inputs images [(3,64,64)]")
flags.DEFINE_integer("image_wh", 64, "Shape of inputs images 64*64")
flags.DEFINE_string("exp_name", None, "The name of experiment [None]")
flags.DEFINE_string("arch", "resnet", "The desired arch: low_cap, high_cap, resnet. [resnet]")
flags.DEFINE_integer("alpha", 0, "alpha value  vector base")
flags.DEFINE_float("beta", 100, "beta value reset 000")
flags.DEFINE_float("lr", 0.001, "learning rate value")
flags.DEFINE_string("output_dir", "./", "Output directory for checkpoints, samples, etc. [.]")
flags.DEFINE_string("data_dir", None, "Data directory [None]")
flags.DEFINE_string("file_ext", ".npz", "Image filename extension [.jpeg]")
flags.DEFINE_boolean("train", True, "Train [True]")
flags.DEFINE_boolean("save_codes", True, "Save latent representation or code for all data samples [False]")
flags.DEFINE_boolean("visualize_reconstruct", True, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.exp_name is None:
        FLAGS.exp_name = "separateRep_{}_{}_{}_lr_{}".format( FLAGS.arch, FLAGS.latent_dim, FLAGS.image_wh, FLAGS.lr)
    image_shape = [int(i) for i in FLAGS.image_shape.strip('()[]{}').split(',')]
    dirs = init_directories(FLAGS.exp_name, FLAGS.output_dir)
    dirs['data'] = '../../../npz_datas' if FLAGS.data_dir is None else FLAGS.data_dir
    dirs['codes'] = os.path.join(dirs['data'], 'codes/')
    create_directories(dirs, FLAGS.train, FLAGS.save_codes)

    output_dim = reduce(mul, image_shape, 1)

    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.allow_growth = True
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=run_config)

    ae = AE(
        session=sess,
        arch=FLAGS.arch,
        lr=FLAGS.lr,
        alpha=FLAGS.alpha,
        beta=FLAGS.beta,
        latent_dim=FLAGS.latent_dim,
        latent_num=FLAGS.latent_num,
        class_net_unit_num=FLAGS.class_net_unit_num,
        output_dim=output_dim,
        batch_size=FLAGS.batch_size,
        image_shape=image_shape,
        exp_name=FLAGS.exp_name,
        dirs=dirs,
        vis_reconst=FLAGS.visualize_reconstruct,
    )


    if FLAGS.visualize_reconstruct:
        sampleNum =1280 # 20x64 large batch, forward prop only
        dataVisualName1='pattern_(20x64)x64x64x3_unitLength9_test_visualdata1'
        dataVisualName2='pattern_(20x64)x64x64x3_unitLength9_test_visualdata2'
        
        data_manager = TeapotsDataManager(dirs['data'],
                        dataVisualName1,dataVisualName2, FLAGS.batch_size, 
                        image_shape, shuffle=False,file_ext=FLAGS.file_ext, train_fract=0.8, 
                        inf=True,supervised=False)  
        #data_manager.set_divisor_batch_size()

        #ae.train_iter, ae.dev_iter, ae.test_iter= data_manager.get_iterators()
        ae.train_iter1, ae.dev_iter1, ae.test_iter1,ae.train_iter2, ae.dev_iter2, ae.test_iter2= data_manager.get_iterators()
        
        ae.session.run(tf.global_variables_initializer())
        # change the which iteration ckpt you want to use
        saved_step = ae.load_fixedNum(inter_num=5750)
        assert saved_step > 1, "A trained model is needed to encode the data!"
        
        pathForSave='VisualImgsResults'
        try:
            os.makedirs(pathForSave)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(pathForSave):
                pass
            else:
                raise

        for batch_num in range(int(sampleNum/FLAGS.batch_size)):
            img_batch1, _mask1, _ = next(ae.train_iter1)
            img_batch2, _mask2, _ = next(ae.train_iter2)
            #code = ae.encode(img_batch) #[batch_size, reg_latent_dim]
            ae.getVisualImgs(pathForSave,img_batch1, _mask1,img_batch2, _mask2,batch_num)
            if batch_num < 5 or batch_num % 10 == 0:
                print(("Batch number {0}".format(batch_num)))
           
        print("Images and Codes saved to Folder: VisualImgsResults")


if __name__ == '__main__':
    tf.app.run()
