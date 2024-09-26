import omas
import database.hsds_shot as hsds_shot

def check_thompson(ods):
    if 'thomson_scattering' not in ods.keys():
        return False
    
    if 'time' not in ods['thomson_scattering'].keys():
        return False
    
    if 'ne.data' not in ods['thomson_scattering'].keys():
        return False
    
    if 'te.data' not in ods['thomson_scattering'].keys():
        return False
    
    return True

def check_equilibrium(ods):
    if 'equilibrium' not in ods.keys():
        return False
    
    if 'time' not in ods['equilibrium'].keys():
        return False
    
    return True
