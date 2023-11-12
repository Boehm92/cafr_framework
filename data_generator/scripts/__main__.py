import madcad as mdc
import pandas as pd
from cube_transform import *
import machining_feature_transform as mft

cad_directory = 'TRAINING_DATASET_SOURCE'

machining_feature_limit = 0.6  # Percentage of the max possible dimension for every machining feature

for i in range(66283, 144001):

    print("Part: ", i)
    label_list = []

    try:
        _cube = mdc.brick(width=mdc.vec3(10))
        _cube = _cube.transform(mdc.vec3(5, 5, 5))
        machining_feature = np.random.randint(0, 24)
        print("machining feature: ", machining_feature)
        model = mft.MachiningFeature(_cube, machining_feature, machining_feature_limit).apply_feature()
        model = rotate_model_randomly(model)
        label_list.append([0, 0, 0, 0, 0, 0, machining_feature])
    except:
        print(" machining feature not feasible")

    number_machining_features = np.random.randint(0, 8)
    for count in range(number_machining_features):
        try:
            additional_machining_feature = np.random.randint(0, 24)
            print("machining_feature: ", additional_machining_feature)

            model = mft.MachiningFeature(model, additional_machining_feature,
                                         machining_feature_limit).apply_feature()
            model = rotate_model_randomly(model)
            label_list.append([0, 0, 0, 0, 0, 0, additional_machining_feature])
        except:
            print("machining feature not feasible")

    # # DA: Random Scale
    # model_scale_factor = np.random.uniform(0.5, 1)
    # model = model.transform(mdc.mat3(model_scale_factor, model_scale_factor, model_scale_factor))

    mdc.write(model, os.getenv(cad_directory) + "/" + str(
        i) + ".stl")  # str(machining_feature) + "_" +
    labels = pd.DataFrame(label_list)
    labels.to_csv(
        os.getenv(cad_directory) + "/" + str(i)  # str(machining_feature) + "_" +
        + ".csv", header=False, index=False)

    del model
    del labels
