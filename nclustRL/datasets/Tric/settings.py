
BINARY_BASE = {
            'shape': [[90, 9], [110, 11]],
            'n': 10,
            'clusters': [10, 10],
            'dataset_settings': {
                'patterns': {
                    'value': [
                        [['CONSTANT', 'CONSTANT'], ['Additive', 'Additive']],
                        [['Additive', 'Constant'], ['CONSTANT', 'CONSTANT']]
                    ],
                    'type': 'categorical',
                    'randomize': True
                },
                'realval': {
                    'value': [True, False],
                    'type': 'categorical',
                    'randomize': True
                },
                'maxval': {
                    'value': 11.0,
                },
                'minval': {
                    'value': [-10.0, 1.0],
                    'type': 'continuous',
                    'randomize': True
                }
            },
            'max_steps': 150
        }

BINARY_N_LESS = {}

BINARY_N_MORE = {}

BINARY_N = {}

BINARY_SHAPES_01 = {}

BINARY_SHAPES_02 = {}

BINARY_SHAPES = {}

BINARY_CLUST_DIM_01 = {}

BINARY_CLUST_DIM_02 = {}

BINARY_CLUST_DIM = {}

BINARY_OVERLAPPING_01 = {}

BINARY_OVERLAPPING_02 = {}

BINARY_OVERLAPPING = {}

BINARY_QUALITY_01 = {}

BINARY_QUALITY_02 = {}

BINARY_QUALITY = {}