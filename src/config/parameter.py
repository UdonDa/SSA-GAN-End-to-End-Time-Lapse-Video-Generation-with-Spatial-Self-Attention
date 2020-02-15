import os
import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters(name='test'):
    parser = argparse.ArgumentParser()

    root = os.path.join(os.getcwd(), 'results')

    parser.add_argument('--mode', type=str, default='train', help='train | save_img | save_val_img')

    parser.add_argument('--attention_type', type=str, default="HW", help = 'HW | THW | noGammaHW | noGammaTHW')

    parser.add_argument('--c_cal', type=int, default=1)    
    parser.add_argument('--my_model', type=str2bool, default=False)
    parser.add_argument('--pretrained_model_path', type=str, default=f'{root}/7_20_MD_S1/models/60-BaseG.ckpt')

    parser.add_argument('--lambda_cycle', type=float, default=10.00, help='for adv ranking loss.')
    parser.add_argument('--lambda_gp', type=float, default=10.00, help='weight for gradient penalty.')
    parser.add_argument('--lambda_triplet', type=float, default=1.00, help='weight for gradient penalty.')

    # Training setting
    parser.add_argument('--total_epochs', type=int, default=100, help='how many times to update the generator')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_critic', type=int, default=5)

    parser.add_argument('--stage1_resume_iter', type=int, default=30)
    parser.add_argument('--stage2_resume_iter', type=int, default=0)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--resume_iters', type=int, default=0)

    # Dataset.
    ### Sky
    parser.add_argument('--data_train', type=str, default='/home/yanai-lab/horita-d/export/dataset/sky_timelapse/sky_train')
    parser.add_argument('--data_test', type=str, default='/home/yanai-lab/horita-d/export/dataset/sky_timelapse/sky_test')

    parser.add_argument('--nframes', type=int, default=32, help='number of frames in each video clip') ### For clowd

    parser.add_argument('--dataset', type=str, default='cloud')
    parser.add_argument('--image_size', type=int, default=128)
    
    parser.add_argument('--channel_cat', type=str2bool, default=True)

    parser.add_argument('--stage1', type=str2bool, default=False)
    parser.add_argument('--stage2', type=str2bool, default=True)
    parser.add_argument('--use_spectral_norm_G', type=str2bool, default=True)
    parser.add_argument('--use_spectral_norm_D', type=str2bool, default=True)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9) #default: 0.9
    parser.add_argument('--g_lr', type=float, default=0.0002) #default: 0.0002
    parser.add_argument('--d_lr', type=float, default=0.0002) #default: 0.0002

    parser.add_argument('--exp_name', type=str, default=name)

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=2)
    parser.add_argument('--model_save_step', type=float, default=2)
    parser.add_argument('--eval_step', type=float, default=1)


    own_stage_1_epoch = 45
    parser.add_argument('--own_stage_1', 
      default=f'{root}/cloud_stage1/{own_stage_1_epoch}-BaseG.ckpt'
      )
    
    own_stage_2_epoch = 12
    parser.add_argument('--own_stage_2',
      default=f'{root}/cloud_stage2/models/{own_stage_2_epoch}-RefineG.ckpt'
      )

    parser.add_argument('--md_stage_1',
      default='ms_s1_030.pth'
      )

    parser.add_argument('--md_stage_2',
      default='ms_s2_067.pth'
      )

    parser.add_argument('--n',
      default=""
      )

    args = parser.parse_args()

    ## Directories.
    args.main_path = f'{root}/{args.exp_name}'
    args.gif_path = f'{root}/{args.exp_name}/gifs'
    args.image_path = f'{root}/{args.exp_name}/images'
    args.log_path =  f'{root}/{args.exp_name}/logs'
    args.model_save_path = f'{root}/{args.exp_name}/models'
    args.sample_path = f'{root}/{args.exp_name}/samples'
    args.val_path = f'{root}/{args.exp_name}/val'
    args.metric_path = f'{root}/{args.exp_name}/metric'
    args.test_generated_path = f'{root}/{args.exp_name}/generated'

    return args