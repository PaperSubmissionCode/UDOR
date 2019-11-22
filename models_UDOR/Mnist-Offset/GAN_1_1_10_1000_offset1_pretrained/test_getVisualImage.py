import os
import errno
import tensorflow as tf
from operator import mul
from functools import reduce
import sys
sys.path.append("../")


from model import AE
from lib.utils import init_directories, create_directories
from lib.models.data_managers_ae import ShapesDataManager

flags = tf.app.flags
flags.DEFINE_integer("No",19, "The No of the test")
flags.DEFINE_integer("epochs",30, "Number of epochs to train [25]")
flags.DEFINE_integer("stats_interval", 1, "Print/log stats every [stats_interval] epochs. [1.0]")
flags.DEFINE_integer("ckpt_interval", 1, "Save checkpoint every [ckpt_interval] epochs. [10]")
flags.DEFINE_integer("latent_dim", 2*1, "Number of latent variables [10]")
flags.DEFINE_integer("latent_num", 2, "Number of latent part")
flags.DEFINE_integer("class_net_unit_num", 9, "Number of neuro cell")
flags.DEFINE_integer("batch_size", 64, "The size of training batches [64]")
flags.DEFINE_string("image_shape", "(1,32,32)", "Shape of inputs images [(1,32,32)]")
flags.DEFINE_integer("image_wh", 32, "Shape of inputs images 64*64")
flags.DEFINE_string("exp_name", None, "The name of experiment [None]")
flags.DEFINE_string("arch", "resnet", "The desired arch: low_cap, high_cap, resnet. [resnet]")
flags.DEFINE_integer("alpha", 0, "alpha value  vector base")
flags.DEFINE_float("beta",100, "beta value reset 000")
flags.DEFINE_float("lr", 0.0005, "ratio value")
flags.DEFINE_string("output_dir", "./", "Output directory for checkpoints, samples, etc. [.]")
flags.DEFINE_string("data_dir", None, "Data directory [None]")
flags.DEFINE_string("file_ext", ".npz", "Image filename extension [.jpeg]")
flags.DEFINE_boolean("gaps", False, "Create gaps in data to faciliate zero-shot inference [False]")
flags.DEFINE_boolean("train", True, "Train [True]")
flags.DEFINE_boolean("save_codes", True, "Save latent representation or code for all data samples [False]")
flags.DEFINE_boolean("visualize_reconstruct", True, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.exp_name is None:
        FLAGS.exp_name = "reconstrued_results"
    image_shape = [int(i) for i in FLAGS.image_shape.strip('()[]{}').split(',')]
    dirs = init_directories(FLAGS.exp_name, FLAGS.output_dir)
    dirs['data'] = '../../../npz_datas_single_test' if FLAGS.data_dir is None else FLAGS.data_dir
    dirs['codes'] = os.path.join(dirs['data'], 'codes/')
    create_directories(dirs, FLAGS.train, FLAGS.save_codes)
    
    output_dim  = reduce(mul, image_shape, 1)
    
    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.allow_growth=True
    run_config.gpu_options.per_process_gpu_memory_fraction=0.9
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
    
    # test get changed images 
    data1Name="mnistOffset1_Imgs_GTimages_mask_GTlabel_(32x64)x32x32x1_unitLength1_CodeImageDataset"
    
    data_manager = ShapesDataManager(dirs['data'],
                    data1Name,FLAGS.batch_size, 
                    image_shape, shuffle=False,file_ext=FLAGS.file_ext, train_fract=1.0, 
                    inf=True)
    ae.train_iter1, ae.dev_iter1, ae.test_iter1= data_manager.get_iterators()

    ae.session.run(tf.global_variables_initializer())
    saved_step = ae.load_fixedNum(1875)
    assert saved_step > 1, "A trained model is needed to encode the data!"
    
    pathForSave='VisualImgsResults'
    try:
        os.makedirs(pathForSave)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(pathForSave):
            pass
        else:
            raise

    assert saved_step > 1, "A trained model is needed to encode the data!"
    sampleNum=2048
    for k in range(int(sampleNum / FLAGS.batch_size)):
        fixed_x1, fixed_mk1 , _ = next(ae.train_iter1)
        #print(fixed_x1.shape)
        #print(fixed_mk1 )
        ae.visualise_reconstruction_path(pathForSave,fixed_x1, fixed_mk1,k)
    #print(k)
    print('finish encode!')



if __name__ == '__main__':
    tf.app.run()
