# data of tasks

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
	    "slope": 0.0025,
	    "max_speed": 0.07,
	    "power" :0.001,
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
	},

	"full_energy" : {
	    "file_name" : "full.p",
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
	},
}