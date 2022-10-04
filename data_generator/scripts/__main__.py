import madcad as mdc
import pandas as pd
from cube_transform import *
import machining_feature_transform as mft

min_scale = 1
max_scale = 9
min_depth = 1
max_depth = 9
cube = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Cube.stl')
cube.mergeclose()
cube = mdc.segmentation(cube)

main_mf = 2
loop_number = 0

for i in range(loop_number, loop_number + 200):
    # if (i % 564) == 0:
    #     main_mf += 1

    main_mf = 0 # np.random.randint(0, 24)

    print("Part: ", i)
    print("Main Feature: ", main_mf)
    label_list = []
    try:
        mf = np.random.randint(0, 24)
        model = mft.MachiningFeature(cube, main_mf, min_scale, max_scale, min_depth, max_depth).apply_feature()
        model = rotate_model_randomly(model)
        label_list.append([0, 0, 0, 0, 0, 0, main_mf])
    except:
        print("MF not feasible")

    number_machining_features = np.random.randint(0, 10)

    for mf in range(number_machining_features):
        try:
            additional_mf = np.random.randint(0, 24)
            print("Additional Feature: ", additional_mf)

            model = mft.MachiningFeature(
                 model, additional_mf, min_scale, max_scale, min_depth, max_depth).apply_feature()
            model = rotate_model_randomly(model)

            label_list.append([0, 0, 0, 0, 0, 0, additional_mf])
        except:
            print(" AF not feasible")

    # # DA: Random Scale
    # model_scale_factor = np.random.uniform(0.5, 1)
    # model = model.transform(mdc.mat3(model_scale_factor, model_scale_factor, model_scale_factor))

    mdc.write(model, os.getenv('TEST_DATASET_SOURCE') + "/" + str(main_mf) + "/" + str(i) + ".stl")

    labels = pd.DataFrame(label_list)
    labels.to_csv(os.getenv('TEST_DATASET_SOURCE') + "/" + str(main_mf) + "/" + str(i) + ".csv", header=False, index=False)
    print(os.getenv('TEST_DATASET_SOURCE') + "/" + str(main_mf) + "/" + str(i))

    del model
    del labels