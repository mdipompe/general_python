'''
Basic transformations between cartesian, polar, and spherical coordinates.
If degree is True, inputs are assumed to be in degrees and outputs will be as well
'''
import numpy as np

def cart2pol(x, y, degree=False):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    if degree:
        theta = theta * (180./np.pi)
    return theta, rho

def pol2cart(theta, rho, degree=False):
    if degree:
        theta = theta * (np.pi/180.)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def cart2sph(x, y, z, degree=False):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(z, hxy)
    phi = np.arctan2(y, x)
    if degree:
        theta = theta * (180./np.pi)
        phi = phi * (180./np.pi)
    return r, theta, phi

def sph2cart(r, theta, phi, degree=False):
    if degree:
        phi = phi * (np.pi/180.)
        theta = theta * (np.pi/180.)
    rcos_theta = r * np.cos(theta)
    x = rcos_theta * np.cos(phi)
    y = rcos_theta * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

def cart2cyl(x, y, z, degree=False):
    rho = np.hypot(x, y)
    if x == 0 & y == 0:
        phi = 0
    if x == 0 & y != 0:
        phi = np.arcsin2(y,rho)
    if x > 0:
        phi = np.arctan2(y,x)
    if x < 0:
        phi = (-1.) * arcsin2(y,rho) + np.pi
    if degree:
        phi = phi * (180./np.pi)
    return rho, phi, z

def cyl2cart(rho, phi, z, degree=False):
    if degree:
        phi = phi * (np.pi/180.)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

def cyl2sph(rho, phi, z, degree=False):
    if degree:
        phi = phi * (np.pi/180.)
    r = np.hypot(rho,z)
    theta = np.arctan2(z,rho)
    if degree:
        theta = theta * (180./np.pi)
    return r, theta, phi

def sph2cyl(r, theta, phi, degree=False):
    if degree:
        theta = theta * (np.pi/180.)
        phi = phi * (np.pi/180.)
    rho = r * np.cos(theta)
    z = r * np.sin(theta)
    return rho, phi, z
        
    
    
