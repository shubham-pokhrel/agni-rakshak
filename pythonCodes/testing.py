
import numpy as np

servo_x,servo_y=640,360

def map_value(fire_pixel, min_pixel, max_pixel, min_target=100, max_target=23):
    mapped_value = np.interp(fire_pixel, (min_pixel, max_pixel), (min_target, max_target))
    return mapped_value

#vertical mapping
def map_value_vertical(fire_pixel, min_pixel, max_pixel, min_target=50, max_target=102):
    mapped_value = np.interp(fire_pixel, (min_pixel, max_pixel), (min_target, max_target))
    return mapped_value






count = 0

# Define the condition for the while loop
while count < 5:
    center_x = int(input("resolution x  "))
    center_y = int(input("resolution y  "))

    
    x,y=center_x-servo_x,center_y-servo_y
    duty_x=map_value(x,-640,640)
    duty_y=map_value_vertical(y,-360,360)

    print(int(duty_x))
    print(int(duty_y))
    print("------------------------------------")
    
    count=count+1

