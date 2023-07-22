import madcad as mdc
import pandas as pd
from cube_transform import *
import machining_feature_transform as mft

machining_feature = 19

cad_directory = 'TRAINING_DATASET_SOURCE'

for i in range(0,  20):
    #
    # if i % 1 == 0:
    #     machining_feature += 1

    print("Part: ", i)
    label_list = []

    try:
        print("machining feature: ", machining_feature)

        model = mft.MachiningFeature(machining_feature).apply_feature()
        # model = rotate_model_randomly(model)
        label_list.append([0, 0, 0, 0, 0, 0, machining_feature])
    except:
         print(" machining feature not feasible")


    # number_machining_features = np.random.randint(0, 10)
    # for count in range(number_machining_features):
    #     try:
    #         additional_machining_feature = np.random.randint(0, 24)
    #         print("additional_machining_feature: ", additional_machining_feature)
    #
    #         model = mft.MachiningFeature(
    #             model, additional_machining_feature, min_scale, max_scale, min_depth, max_depth).apply_feature()
    #         model = rotate_model_randomly(model)
    #
    #         label_list.append([0, 0, 0, 0, 0, 0, additional_machining_feature])
    #     except:
    #         print(" machining feature not feasible")

    # # DA: Random Scale
    # model_scale_factor = np.random.uniform(0.5, 1)
    # model = model.transform(mdc.mat3(model_scale_factor, model_scale_factor, model_scale_factor))

    mdc.write(model, os.getenv(cad_directory) + "/" + str(i) + ".stl")
    labels = pd.DataFrame(label_list)
    labels.to_csv(os.getenv(cad_directory) + "/" + str(i) + ".csv", header=False, index=False)

    del model
    del labels
