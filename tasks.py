import numpy as np

SOURCES = {
    "standard": {
        "reward_modes": [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0),
                ("still", 0),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power": 0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles": [] 
    },

    "more_actions" : {
        "reward_modes" : [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0)
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "obstacles" : [] ,
        "neutral": 2, 
        "actions_nb" : 5
    },

    "bananas": {
        "reward_modes": [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0),
                ("still", 0),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power": 0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0.1,
        "h_bananas": 0.2,
        "obstacles": [] 
    },

    "bumps" : {
        "reward_modes": [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0),
                ("still", 0),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power": 0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [
        (-.5, .3, .3), 
        (.3, .2, .1), 
        (0, .2, .1)] 
    },

    "easy_slope" : {
        "reward_modes": [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0),
                ("still", 0),
                ],
        "slope": 0.0015,
        "max_speed": 0.07,
        "power": 0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [] 
    },

    "more_power" : {
        "reward_modes": [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0),
                ("still", 0),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power": 0.005,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [] 
    },

    "energy" : {
        "reward_modes": [
                ("time", 0),
                ("energy", 1),
                ("distance", 0),
                ("center", 0),
                ("height", 0),
                ("still", 0),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power": 0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [] 
    },

    "distance" : {
        "reward_modes" : [
                ("time", 0),
                ("energy", 0),
                ("distance", 1),
                ("center", 0),
                ("height", 0),
                ("still", 0),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [] 
    },
    "center" : {
        "reward_modes" : [
                ("time", 0),
                ("energy", 0),
                ("distance", 0),
                ("center", 1),
                ("height", 0),
                ("still", 0),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [] 
    }, 
    "height" : {
        "reward_modes" : [
                ("time", 0),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 1),
                ("still", 0),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [] 
    },   
    "still" : {
        "reward_modes" : [
                ("time", 0),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0),
                ("still", 1),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [] 
    },     
}


TARGETS = {
    "full_clean" : {
        "reward_modes" : [
                ("time", 1),
                ("energy", 1),
                ("distance", 1),
                ("center", 1),
                ("height", 1),
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [] 

    },

    "full" : {
        "reward_modes" : [
                ("time", 1),
                ("energy", 1),
                ("distance", 1),
                ("center", 1),
                ("height", 1)
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "min_position": -1.2,
        "low": -0.6,
        "high": -0.4,
        "actions_nb": 3,
        "neutral": 1,
        "p_bananas": 0,
        "h_bananas": 0.1,
        "obstacles" : [
        (-.7, .1, .1), 
        (.3, .3, .3), 
        (1., .1, .2)] 
    },

    "more_actions" : {
        "reward_modes" : [
                ("time", 1),
                ("energy", 1),
                ("distance", 1),
                ("center", 1),
                ("height", 1)
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "obstacles" : [
        (-.5, .3, .3), 
        (.3, .2, .1), 
        (0, .2, .1)] ,
        "actions_nb" : 5,
        "neutral": 2
    },

    "full_energy" : {
        "reward_modes" : [
                ("time", 1),
                ("energy", 100),
                ("distance", 1),
                ("center", 1),
                ("height", 1)
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "obstacles" : [
        (-.5, .3, .3), 
        (.3, .2, .1), 
        (0, .2, .1)] 
    },
}