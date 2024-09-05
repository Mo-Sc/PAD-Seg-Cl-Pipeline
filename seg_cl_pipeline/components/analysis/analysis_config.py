class ClassificationConfigDemo:
    """
    Config for Demo Data
    """

    target_scan = 1

    chromo = "mSO2"  # has to be same as loaded OA type
    invert = True if chromo == "Hb" else False

    # in case of splitting into derivation / valiation set (not used)
    de_val_split = False
    random_state = 42
    n_splits = 5
    test_size = 0.2

    # just for visualization
    width = 400
    height = 400
    title = "Demo: ROC Curve HV - CLTI"


class ClassificationConfigDatasetA:
    """
    Config for PAD Classification on Dataset A
    """

    target_scan = 9

    chromo = "mSO2"  # has to be same as loaded OA type
    invert = True if chromo == "Hb" else False

    # in case of splitting into derivation / valiation set (not used)
    de_val_split = False
    random_state = 42
    n_splits = 5
    test_size = 0.2

    # just for visualization
    width = 400
    height = 400
    title = "Dataset A: ROC Curve HV - IC"


class ClassificationConfigDatasetB:
    """
    Config for PAD Classification on Dataset B
    """

    target_scan = 1

    chromo = "HbO2"  # has to be same as loaded OA type
    invert = True if chromo == "Hb" else False

    # in case of splitting into derivation / valiation set (not used)
    de_val_split = False
    random_state = 42
    n_splits = 5
    test_size = 0.2

    # just for visualization
    width = 400
    height = 400
    title = "Dataset B: ROC Curve HV - CLTI"
