"""Dataset configurations."""
import config
from PIL import Image
"""Dataset configurations."""
import pwd
from os.path import join, isdir, isfile
from torchvision import transforms
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class DatasetConfig():
    """Dataset configurations."""

    #父目录路径
    # data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../MMD/dataset"))
    data_directory = '/home/student2020/MMD/dataset'
    # data_directory = join(data_directory, 'dataset')

    # dialog_directory = join(data_directory, 'MMD2withDST_dst_cleaned_only')
    dialog_directory = data_directory

    # url2img
    url2img = join(data_directory, 'url2img.txt')
    # image_id_file
    image_id_file = join(data_directory, 'cleaned_image_id.json')

    # dataset path
    train_dialog_data_directory = join(dialog_directory, 'MMD2withDST/train/')
    valid_dialog_data_directory = join(dialog_directory, 'MMD2withDST/valid/')
    test_dialog_data_directory = join(dialog_directory, 'MMD2withDST/test/')

    wodst_train_dialog_data_directory = join(dialog_directory, 'MMD2woDST/train/')
    wodst_fewshot_dialog_data_directory = join(dialog_directory, 'MMD2woDST/fewshot/')
    wodst_valid_dialog_data_directory = join(dialog_directory, 'MMD2woDST/valid/')
    wodst_test_dialog_data_directory = join(dialog_directory, 'MMD2woDST/test/')

    # MMD_material_path = join(data_directory, '../MMD/dataset')
    MMD_material_path = data_directory

    # image path
    image_data_directory = join(MMD_material_path, 'images/images')

    # image attrs path
    product_data_directory = join(MMD_material_path, 'knowledge/products_format/')

    # dump_dir = join(data_directory, '../../shy/MIC-master/mmd_data')
    dump_dir = join(data_directory, '../../shy/MIC-master/mmd_data')

    image_url_id_dir = join(dump_dir, 'img_url_id_data.pkl')
    img_path_dir = join(dump_dir, 'img_path_data.pkl')
    image_text_dir = join(dump_dir, 'image_text_data.pkl')
    image_taxonomy_dir = join(dump_dir, 'image_taxonomy_data.pkl')
    new_image_taxonomy_dir = join(dump_dir, 'new_image_taxonomy_data.pkl')
    stage1_model_dir = join(dump_dir, 'stage1_model.pt')
    stage2_model_dir = join(dump_dir, 'stage2_model.pt')

    common_raw_data_file = join(dump_dir, 'common_raw_data.pkl')

    wodst_common_raw_data_file = join(dump_dir, 'wodst_common_raw_data.pkl')

    train_raw_data_file = join(dump_dir, 'train_raw_data.pkl')
    fewshot_raw_data_file = join(dump_dir, 'fewshot_raw_data.pkl')
    valid_raw_data_file = join(dump_dir, 'valid_raw_data.pkl')
    test_raw_data_file = join(dump_dir, 'test_raw_data.pkl')

    wodst_train_raw_data_file = join(dump_dir, 'wodst_train_raw_data.pkl')
    wodst_valid_raw_data_file = join(dump_dir, 'wodst_valid_raw_data.pkl')
    wodst_test_raw_data_file = join(dump_dir, 'wodst_test_raw_data.pkl')

    img_feature_path = join(dump_dir, 'extracted_img_feature.pkl')

    recommend_train_dialog_file = join(dump_dir,
                                       'recommend_train_dialog_file.pkl')
    recommend_fewshot_dialog_file = join(dump_dir,
                                         'recommend_fewshot_dialog_file.pkl')
    recommend_valid_dialog_file = join(dump_dir,
                                       'recommend_valid_dialog_file.pkl')
    recommend_test_dialog_file = join(dump_dir,
                                      'recommend_test_dialog_file.pkl')

    wodst_recommend_train_dialog_file = join(dump_dir,
                                             'wodst_recommend_train_dialog_file.pkl')
    wodst_recommend_valid_dialog_file = join(dump_dir,
                                             'wodst_recommend_valid_dialog_file.pkl')
    wodst_recommend_test_dialog_file = join(dump_dir,
                                            'wodst_recommend_test_dialog_file.pkl')

    special_tokens = 'special_tokens.json'

    tensorboard_file = 'tensorboard/'

    dialog_context_size = 5
    with_img_dialog_context_size = 2

    text_max_len = 512