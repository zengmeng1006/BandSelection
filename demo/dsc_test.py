from BandSelection.classes.DSC_NET import DSCBS
from BandSelection.Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from BandSelection.classes.utility import eval_band



# root = 'F:\Python\HSI_Files\\'input channels does not match filter's input channels, 10 != 32
root = '/Users/cengmeng/PycharmProjects/python/Deep-subspace-clustering-networks/Data/'

# im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
# im_, gt_ = 'Pavia', 'Pavia_gt'
# im_, gt_ = 'Botswana', 'Botswana_gt'
# im_, gt_ = 'KSC', 'KSC_gt'

img_path = root + im_ + '.mat'
gt_path = root + gt_ + '.mat'
print(img_path)

p = Processor()
img, gt = p.prepare_data(img_path, gt_path)
# Img, Label = Img[:256, :, :], Label[:256, :]
n_row, n_column, n_band = img.shape
X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
img_correct, gt_correct = p.get_correct(X_img, gt)
train_inx, test_idx = p.get_tr_tx_index(gt_correct, test_size=0.4)

n_input = [n_row, n_column]
kernel_size = [32,10]
n_hidden = [32,10]
batch_size = n_band
model_path = './pretrain-model-COIL20/model.ckpt'
ft_path = './pretrain-model-COIL20/model.ckpt'
logs_path = './pretrain-model-COIL20/logs'

batch_size_test = n_band

iter_ft = 0
display_step = 1
alpha = 0.04
learning_rate = 1e-3

reg1 = 1e-4
reg2 = 150.0

n_selected_band = [5, 10, 15, 20, 25]
n_iter = 50
# kwargs = {'n_input': n_input, 'n_hidden': n_hidden, 'reg_const1': reg1, 'reg_const2': reg2, 'max_iter':50,
#               'kernel_size': kernel_size, 'batch_size': batch_size_test, 'model_path': model_path,
#               'logs_path': logs_path}
# iter = []
# score = []
# for i in range(n_iter):
#     if i-50 % 50 == 0:
kwargs = {'n_input': n_input, 'n_hidden': n_hidden, 'reg_const1': reg1, 'reg_const2': reg2, 'max_iter': n_iter,
              'kernel_size': kernel_size, 'batch_size': batch_size_test, 'model_path': model_path,
              'logs_path': logs_path}

for i in n_selected_band:
    X_new = DSCBS(n_selected_band, **kwargs).predict(X_img)
    X_new, _ = p.get_correct(X_new, gt)
    acc = eval_band(X_new, gt_correct, train_inx, test_idx)
    print('i=%d, acc=%s' % i, acc)
        # iter.append(i)
        # score.append(acc)

# plt.figure()
# plt.plot(iter, score, 'r--')
# plt.xlabel('iter')
# plt.ylabel('score')
# plt.show()

