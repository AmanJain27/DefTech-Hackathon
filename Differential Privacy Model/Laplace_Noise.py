from opendp.mod import enable_features
enable_features("contrib")
enable_features("floating-point")

from opendp.typing import VectorDomain, AllDomain, SymmetricDistance

import opendp.trans as tf
from opendp.meas import make_base_laplace, make_base_geometric
from opendp.comb import make_chain_mt
import math 
# First order of business, get the data
# just read the csv file output 

def getData(path):
    with open(path) as f:
        data = f.read()

    return data

#function: rounded
#parameter: number of type float
#return val: nearest integer rounded
def rounded(number):
    if number-int(number) > 0.5:
        return math.ceil(number)
    return math.floor(number)

# function: Transformation
# return val: count of the tuples
# description: This function creates the transformation on the data to interpret
# csv files and apply appropriate casting. Then using the transformation, the
# count of the tuples in that particular data is returned
def CSVTransformation(data, col_names, key_col):
    preprocessor = (
            tf.make_split_dataframe(',', col_names=col_names) >>
            tf.make_select_column(key=key_col, TOA=str) >> 
            tf.make_cast_default(TIA=str, TOA=int)
            )
    count = preprocessor >> tf.make_count(TIA=int)
    return count 

def Transformation(n_node_samples, epsilon):
    laplace_base = make_base_laplace(scale=float(epsilon))
   
    release = []
    for i in n_node_samples:
        number = laplace_base(i)
        number = rounded(number)
        release.append(abs(number)) 
    return release
    

# function: LaplaceTransformation
# return val: the value + random noise
# since the noise output is through laplace, the bounds defined here are floats
# and not integers as make_clamp would expect
def LaplaceTransformation(data, col_names, key_col, bounds, epsilon):
    bounds = tuple(map(float, bounds))
    count = Transformation(data, col_namas, key_col)
    preprocessor = ( 
            tf.make_split_dataframe(',', col_names=col_names) >>
            tf.make_select_column(key=key_col, TOA=str) >>
            tf.make_cast_default(TIA=str, TOA=float) >> 
            tf.make_clamp(bounds=bounds) >> 
            tf.make_sized_bounded_sum(size=count, bounds=bounds)
            )
    # use the epsilon value provided and add the noise
    budget = tf.binary_search_chain(
            lambda s: preprocessor >> make_base_laplace(scale=s),
            d_in=1, d_out=epsilon
            )
    return rounded(budget(data)) 

# function: LaplaceNoise
# return val: random noise
# parameters: a number, epsilon value
# desc: takes a number as an input and ouputs the random noise according to 
# epsilon value provided

def LaplaceNoise(number, epsilon):
    
    p = (
        tf.make_identity(D=VectorDomain[AllDomain[int]], M=SymmetricDistance)
        )
    count = p >> tf.make_count(TIA=int)
    print(count(VectorDomain[AllDomain[number]]))    


    
#print(LaplaceNoise(302, 0.01))

