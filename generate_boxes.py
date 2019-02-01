from helpers import *
import pickle

PATH_TO_TEST_IMAGES_DIR = r'/Users/zhenzhenweng/Downloads/images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, f) for f in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
PATH_TO_TEST_LABELS_DIR = r'/Users/zhenzhenweng/Downloads/labels'
TEST_LABEL_PATHS = [os.path.join(PATH_TO_TEST_LABELS_DIR, f) for f in os.listdir(PATH_TO_TEST_LABELS_DIR)]

train_object_names = []
train_object_x = []
train_object_y = []
train_object_height = []
train_object_width = []
vg_idx = {}
ground_truth = {}

print('Total: {} labels.'.format(len(TEST_LABEL_PATHS)))
with detection_graph.as_default():
    with tf.Session() as sess:
        for i, label_path in enumerate(TEST_LABEL_PATHS):
            with open(label_path) as f:
                labels = f.read().strip().split('\n')
                # count number of cyclist in the image. 
                cyclist_count = 0
                for lab in labels:
                    if lab.startswith('Cyclist'):
                        cyclist_count += 1
                ground_truth[i] = cyclist_count

            image_path = label_path.replace('labels', 'images').strip('.txt')+'.png'
            vg_idx[i] = image_path
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            output_dict = run_inference_for_single_image(image_np, sess)
            plot_class = [1, 2, 3]  # person, bike, car 

            min_score_thresh = {1: 0.2, 2: 0, 3: 0.4}
            curr_train_object_names = []
            curr_train_object_x = []
            curr_train_object_y = []
            curr_train_object_height = []
            curr_train_object_width = []
            print('Image {}, {} with {} cyclists.'.format(i, image_path, cyclist_count))
            for c in plot_class:
                boxes = output_dict['detection_boxes'][output_dict['detection_classes'] == c]
                classes = output_dict['detection_classes'][output_dict['detection_classes'] == c]
                scores = output_dict['detection_scores'][output_dict['detection_classes'] == c]
                
                boxes = boxes[scores >= min_score_thresh[c]]
                classes = classes[scores >= min_score_thresh[c]]
                scores = scores[scores >= min_score_thresh[c]]
                print('Found {} classes of type {}'.format(len(boxes), c))
                # b = [y1, x1, y2, x2]. height = y2 - y1; width = x2 - x1. 
                # if y1(human) < y1(bike)
                curr_train_object_names += [category_index[c]['name'] for b in boxes]
                curr_train_object_x += [int(round(b[1] * image.size[0])) for b in boxes]
                curr_train_object_y += [int(round(b[0] * image.size[1])) for b in boxes]
                curr_train_object_height += [int(round((b[2]-b[0]) * image.size[1])) for b in boxes]
                curr_train_object_width += [int(round((b[3]-b[1]) * image.size[0])) for b in boxes]

            train_object_names += [tuple(curr_train_object_names)]
            train_object_x += [tuple(curr_train_object_x)]
            train_object_y += [tuple(curr_train_object_y)]
            train_object_height += [tuple(curr_train_object_height)]
            train_object_width += [tuple(curr_train_object_width)]
            
train_object_names = np.array(train_object_names)
train_object_x = np.array(train_object_x)
train_object_y = np.array(train_object_y)
train_object_height = np.array(train_object_height)
train_object_width = np.array(train_object_width)
        
train_num = int(len(train_object_names)/3*2)
print('Generated {} training samples'.format(train_num))


val_object_names = train_object_names[train_num:]
val_object_x = train_object_x[train_num:]
val_object_y = train_object_y[train_num:]
val_object_height = train_object_height[train_num:]
val_object_width = train_object_width[train_num:]
np.save(r'/Users/zhenzhenweng/Downloads/val_object_names.npy', val_object_names)
np.save(r'/Users/zhenzhenweng/Downloads/val_object_x.npy', val_object_x)
np.save(r'/Users/zhenzhenweng/Downloads/val_object_y.npy', val_object_y)
np.save(r'/Users/zhenzhenweng/Downloads/val_object_height.npy', val_object_height)
np.save(r'/Users/zhenzhenweng/Downloads/val_object_width.npy', val_object_width)



train_object_names = train_object_names[:train_num]
train_object_x = train_object_x[:train_num]
train_object_y = train_object_y[:train_num]
train_object_height = train_object_height[:train_num]
train_object_width = train_object_width[:train_num]
np.save(r'/Users/zhenzhenweng/Downloads/train_object_names.npy', train_object_names)
np.save(r'/Users/zhenzhenweng/Downloads/train_object_x.npy', train_object_x)
np.save(r'/Users/zhenzhenweng/Downloads/train_object_y.npy', train_object_y)
np.save(r'/Users/zhenzhenweng/Downloads/train_object_height.npy', train_object_height)
np.save(r'/Users/zhenzhenweng/Downloads/train_object_width.npy', train_object_width)

with open(r'/Users/zhenzhenweng/Downloads/vg_idx', 'wb') as f:
    pickle.dump(vg_idx, f)

with open(r'/Users/zhenzhenweng/Downloads/ground_truth', 'wb') as f:
    pickle.dump(ground_truth, f)