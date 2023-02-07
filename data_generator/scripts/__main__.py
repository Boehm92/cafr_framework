import madcad as mdc
import pandas as pd
from cube_transform import *
import machining_feature_transform as mft

min_scale = 1
max_scale = 6
min_depth = 1
max_depth = 6

machining_feature = 0

# 74, 1766
# Training: 147 3633 | Test: 22 530 | Training-Val-Seperation: 3072

for i in range(1, 530):

    if i % 22 == 0:
        machining_feature += 1

    print("Part: ", i)
    model = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Cube.stl')
    model.mergeclose()
    model = mdc.segmentation(model)
    label_list = []


    number_machining_features = 1 #np.random.randint(0, 2)
    for count in range(number_machining_features):
        try:
            #machining_feature = np.random.randint(0, 24)
            print("machining feature: ", machining_feature)

            model = mft.MachiningFeature(
                model, machining_feature, min_scale, max_scale, min_depth, max_depth).apply_feature()
            model = rotate_model_randomly(model)

            label_list.append([0, 0, 0, 0, 0, 0, machining_feature])
        except:
            print(" machining feature not feasible")


    number_machining_features = 1  # np.random.randint(0, 2)
    for count in range(number_machining_features):
        try:
            additional_machining_feature = np.random.randint(0, 24)
            print("additional_machining_feature: ", additional_machining_feature)

            model = mft.MachiningFeature(
                model, additional_machining_feature, min_scale, max_scale, min_depth, max_depth).apply_feature()
            model = rotate_model_randomly(model)

            label_list.append([0, 0, 0, 0, 0, 0, additional_machining_feature])
        except:
            print(" machining feature not feasible")

    # # DA: Random Scale
    # model_scale_factor = np.random.uniform(0.5, 1)
    # model = model.transform(mdc.mat3(model_scale_factor, model_scale_factor, model_scale_factor))

    mdc.write(model, os.getenv('TEST_DATASET_SOURCE') + "/" + str(i) + ".stl")
    labels = pd.DataFrame(label_list)
    labels.to_csv(os.getenv('TEST_DATASET_SOURCE') + "/" + str(i) + ".csv", header=False, index=False)

    del model
    del labels
