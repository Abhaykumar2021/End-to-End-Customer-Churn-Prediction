from setuptools import setup, find_packages

def get_requirements(file_path:str)->list[str]:
    '''This function reads a requirements file and returns a list of requirements.'''

    requirements = [] 
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .") 
    return requirements  

setup(
    name = "Customer Churn Prediction",
    version = "0.0.1",
    author = "Abhay Kumar",
    author_email = "lavadon2021@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')    
    )   