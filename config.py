# -------------------------------- MIMIC diag --------------------------------

# teacher model
config_mimic_diag_teacher = {
    
    'model_type': 'teacher',
    'is_debug': False,
    'batch_size': 128,
    
    'len_data': 48,
    'len_split': 12, 
    'n_channel': 6, 
    'n_class': 8, 
    'temperature': 5, 
    
    'use_conv2':False,
        
    'conv':{
        'filters': 64, 
        'kernel_size': 4, 
        'strides': 2, 
    },
    
    'rnn':{
        'hidden_size': 32, 
    }

}

# student model, the architecture is samller than teacher model
# CNN filter, RNN hidden_size is 1/4
config_mimic_diag_student_light = {
    
    'model_type': 'student',
    'is_debug': False,
    'batch_size': 128,
    
    'len_data': 48,
    'len_split': 12, 
    'n_channel': 1, 
    'n_class': 8, 
    'temperature': 5, 
    
    'use_conv2':False,
        
    'conv':{
        'filters': 16, 
        'kernel_size': 4, 
        'strides': 2, 
    },
    
    'rnn':{
        'hidden_size': 8, 
    }

}

# -------------------------------- PAMAP2 --------------------------------

# teacher model
config_pamap_teacher = {
    
    'model_type': 'teacher',
    'is_debug': False,
    'batch_size': 128,
    
    'len_data': 256,
    'len_split': 64, 
    'n_channel': 52, 
    'n_class': 12, 
    'temperature': 5, 
    
    'use_conv2':False,
        
    'conv':{
        'filters': 64, 
        'kernel_size': 8, 
        'strides': 4, 
    },
    
    'rnn':{
        'hidden_size': 32, 
    }

}

# student model, the architecture is samller than teacher model
# CNN filter, RNN hidden_size is 1/4
config_pamap_student_light = {
    
    'model_type': 'student',
    'is_debug': False,
    'batch_size': 128,
    
    'len_data': 256,
    'len_split': 64, 
    'n_channel': 17, 
    'n_class': 12, 
    'temperature': 5, 
    
    'use_conv2':False,
        
    'conv':{
        'filters': 16, 
        'kernel_size': 8, 
        'strides': 4, 
    },
    
    'rnn':{
        'hidden_size': 8, 
    }

}

# -------------------------------- PTBDB --------------------------------

# teacher model
config_ptbdb_teacher = {
    
    'model_type': 'teacher',
    'is_debug': False,
    'batch_size': 128,
    
    'len_data': 2000,
    'len_split': 500, 
    'n_channel': 15, 
    'n_class': 6, 
    'temperature': 5, 
    
    'use_conv2':True,
        
    'conv':{
        'filters': 128, 
        'kernel_size': 32, 
        'strides': 8, 
    },
    
    'rnn':{
        'hidden_size': 64, 
    }

}


# student model, the architecture is samller than teacher model
# CNN filter, RNN hidden_size is 1/4
config_ptbdb_student_light = {
    
    'model_type': 'student',
    'is_debug': False,
    'batch_size': 128,
    
    'len_data': 2000,
    'len_split': 500, 
    'n_channel': 1, 
    'n_class': 6, 
    'temperature': 5, 
    
    'use_conv2':True,
        
    'conv':{
        'filters': 16, 
        'kernel_size': 32, 
        'strides': 8, 
    },
    
    'rnn':{
        'hidden_size': 8, 
    }

}

