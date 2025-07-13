#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Import and Print the numpy version and the configuration (★☆☆)
import numpy as np
np.version.version


# In[5]:


# 2. Create a null vector of size 10 (★☆☆)
ten_zero = np.zeros(10)
print(ten_zero)


# In[8]:


# How to find the memory size of any array (★☆☆)
np.size(ten_zero)


# In[9]:


# Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
ten_zero[5] = 1
print(ten_zero)


# In[12]:


#Create a vector with values ranging from 10 to 49 (★☆☆)
first_r = np.arange(10, 50)
print(first_r)


# In[13]:


# Reverse a vector (first element becomes last) (★☆☆)
revert_first_r = np.flip(first_r) # first_r[::-1]
print(revert_first_r)


# In[14]:


# Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
twoD = np.arange(9).reshape(3, 3)
print(twoD)


# In[19]:


# Find indices of non-zero elements from [1,2,0,0,4,0,5,7,0,0,23,67,0,0,0] (★☆☆)
my_indices = np.where(~np.isin([1,2,0,0,4,0,5,7,0,0,23,67,0,0,0], 0))
print(my_indices)


# In[20]:


#Create a 5x5 identity matrix (★☆☆)
identity_matrix = np.eye(5)
print(identity_matrix)


# In[24]:


# Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
ten_ten = np.random.randint(0, 200, size=(10, 10))
print(ten_ten)
max_ten_ten = np.max(ten_ten)
min_ten_ten = np.min(ten_ten)
print(min_ten_ten, max_ten_ten)


# In[26]:


# Create a random vector of size 30 and find the mean value (★☆☆)
my_arr= np.random.randint(1, 500, size=(30))
mean_of_my_arr = np.mean(my_arr)
print(mean_of_my_arr)


# In[28]:


# Create a 2d array with 1 on the border and 0 inside (★☆☆)
my_arr1 = np.zeros((6, 6))
my_arr1[0, :] = 1   #top
my_arr1[-1, :] = 1  #bottom
my_arr1[:, -1] = 1  #right
my_arr1[:, 0] = 1   #left
print(my_arr1)


# In[33]:


# How to add a border (filled with 0's) around an existing array?
z = np.ones((6, 6))
z = np.pad(z, pad_width=1, mode='constant', constant_values=0)
print(z)


# In[34]:


# What is the result of the following expression? (★☆☆)
# 0 * np.nan  --> nan
# np.nan == np.nan ---> false
# np.inf > np.nan ---> false
# np.nan - np.nan ---> nan
# np.nan in set([np.nan]) ---> true
# 0.3 == 3 * 0.1  ---> false ---> np.isclose(0.3, 3 * 0.1)


# In[42]:


# Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
diag_arr = np.diag([1,2,3,4], k=-1)
print(diag_arr)

'''x = np.arange(9).reshape((3, 3))
print(x)
print(np.diag(x)) # Output: array([0, 4, 8])
print(np.diag(x, k=1)) # Output: array([1, 5])
print(np.diag(x, k=-1)) # Output: array([3, 7])
print(np.diag(np.diag(x))) # Output: array([[0, 0, 0], [0, 4, 0], [0, 0, 8]])'''


# In[45]:


# Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
x = np.zeros((8,8))
x[1::2, ::2] = 1
x[::2, 1::2] = 1
print(x)


# In[49]:


# Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)
x = np.arange(336)
x= x.reshape(6, 7, 8)
index = np.unravel_index(100, x.shape)
print("Index:", index)


# In[51]:


#  Create a checkerboard 8x8 matrix using the tile function (★☆☆)
pattern = np.array([[0,1], [0,1]])
x = np.tile(pattern, (4, 4))
print(x)


# In[54]:


# Normalize a 5x5 random matrix (★☆☆)
matrix = np.random.rand(5, 5)
row_sums = matrix.sum(axis=1, keepdims=True)
normalized_matrix = matrix / row_sums

print("Original Matrix:\n", matrix)
print("Row-wise Normalized Matrix:\n", normalized_matrix)


# In[55]:


# Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
rgba_dtype = np.dtype([
    ('r', np.uint8),  # Red channel
    ('g', np.uint8),  # Green channel
    ('b', np.uint8),  # Blue channel
    ('a', np.uint8)   # Alpha channel
])


colors = np.array([(255, 0, 132, 255), (0, 234, 0, 255), (0, 0, 255, 179)], dtype=rgba_dtype)

# Print the array
print(colors)


# In[58]:


# Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
x = np.random.randint(15, size=(5, 3))
y = np.random.randint(15, size=(3, 2))
result = x @ y
print(result)


# In[61]:


# Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
x = np.random.randint(50, size=20)
print(x)
x[3:9:] *= -1
print(x)


# In[62]:


# Consider an integer vector Z, which of these expressions are legal? (★☆☆)
# Z**Z legal
# 2 << Z >> 2  legal
# Z <- Z nicht legal
# 1j*Z   legal
# Z/1/1  legal
# Z<Z>Z  legal


# In[63]:


# How to round away from zero a float array ? (★☆☆)
arr = np.array([-1.5, -2.3, 0.2, 1.7, 2.5])
rounded = np.copysign(np.ceil(np.abs(arr)), arr)
print(rounded)


# In[66]:


# How to find common values between two arrays? (★☆☆)
array1 = np.array([1, 4, 3, 4, 5])
array2 = np.array([4, 5, 6, 7, 4])
common_values = np.intersect1d(array1, array2)
print("Common values:", common_values)


# In[69]:


# How to get the dates of yesterday, today and tomorrow? (★☆☆)
today = np.datetime64('today', 'D')
yesterday = today - np.timedelta64(1, 'D')
tomorrow = today + np.timedelta64(1, 'D')

print("Yesterday:", yesterday)
print("Today:", today)
print("Tomorrow:", tomorrow)


# In[ ]:




