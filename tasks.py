import numpy as np
# generate tasks


def generate_tasks(n, r=10):
    sources = dict()
    target = dict()
    norm = 0
    time, energy, distance, center, height, slope, max_speed, power = 0, 0, 0, 0, 0, 0, 0, 0
    for i in xrange(n):
        name = "task_{}".format(i)
        v = np.random.random(8)
        t = int(r*v[0])
        e = int(r*v[1])
        d = int(r*v[2])
        c = int(r*v[3])
        h = int(r*v[4])
        s = 0.0015 + 0.0015*v[5]
        m = 0.05 + 0.005*v[6]
        p = 0.001 + 0.009*v[7]
        sources[name] = {
            "file_name" : "{}.p".format(name),
            "reward_modes" : [
                    ("time", t),
                    ("energy", e),
                    ("distance", d),
                    ("center", c),
                    ("height", h)
                    ],
            "slope": s,
            "max_speed": m,
            "power" : p
        }


        prop = np.random.random()
        norm += prop
        time += prop*t
        energy += prop*e
        distance += prop*d
        center += prop*c
        height += prop*h
        slope += prop*s
        max_speed += prop*m
        power += prop*p

    target["target"] = {
            "file_name" : "target.p",
            "reward_modes" : [
                    ("time", time/norm),
                    ("energy", energy/norm),
                    ("distance", distance/norm),
                    ("center", center/norm),
                    ("height", height/norm)
                    ],
            "slope": slope/norm,
            "max_speed": max_speed/norm,
            "power" : power/norm
        }

    return sources, target




SOURCES = {
    "standard" : {
        "file_name" : "standard.p",
        "reward_modes" : [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0)
                ],
        "slope" : 0.0025,
        "max_speed" : 0.07,
        "power" : 0.001,
        "obstacles" : [] 
    },

    "easy slope" : {
        "file_name" : "easy.p",
        "reward_modes" : [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0)
                ],
        "slope": 0.0015,
        "max_speed": 0.07,
        "power" :0.001,
        "obstacles" : [] 
    },

    "more power" : {
        "file_name" : "power.p",
        "reward_modes" : [
                ("time", 1),
                ("energy", 0),
                ("distance", 0),
                ("center", 0),
                ("height", 0)
                ],
        "slope": 0.0015,
        "max_speed": 0.07,
        "power" :0.005,
        "obstacles" : [] 
    },

    "energy" : {
        "file_name" : "energy.p",
        "reward_modes" : [
                ("time", 0),
                ("energy", 1),
                ("distance", 0),
                ("center", 0),
                ("height", 0)
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "obstacles" : [] 
    },

    "distance" : {
        "file_name" : "distance.p",
        "reward_modes" : [
                ("time", 0),
                ("energy", 0),
                ("distance", 1),
                ("center", 0),
                ("height", 0)
                ],
        "slope": 0.0025,
        "max_speed": 0.07,
        "power" :0.001,
        "obstacles" : [] 
    },

    # "center" : {
    #     "file_name" : "center.p",
    #     "reward_modes" : [
    #             ("time", 0),
    #             ("energy", 0),
    #             ("distance", 0),
    #             ("center", 1),
    #             ("height", 0)
    #             ],
    #     "slope": 0.0025,
    #     "max_speed": 0.07,
    #     "power" :0.001,
    # },

    # "height" : {
    #     "file_name" : "height.p",
    #     "reward_modes" : [
    #             ("time", 0),
    #             ("energy", 0),
    #             ("distance", 0),
    #             ("center", 0),
    #             ("height", 1)
    #             ],
    #     "slope": 0.0025,
    #     "max_speed": 0.07,
    #     "power" :0.001,
    # },

    
}


TARGET = {
    "full_clean" : {
        "file_name" : "full.p",
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
        "obstacles" : [] 
    },

    "full" : {
        "file_name" : "full.p",
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
        "obstacles" : [(-.5, .1, .01), (0, .1, .05)] 
    },

    "full_energy" : {
        "file_name" : "full_energy.p",
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
        "obstacles" : [(-.5, .1, .01), (0, .1, .05)] 
    },
}