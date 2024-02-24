import pandas as pd

dataset = pd.read_csv("datasets/car_insurance.csv",sep=",",decimal=".")

dataset.drop(columns=["engine_type"], inplace=True)
dataset.drop(columns=["is_tpms"], inplace=True)
dataset.drop(columns=["is_parking_sensors"], inplace=True)
dataset.drop(columns=["is_parking_camera"], inplace=True)
dataset.drop(columns=["rear_brakes_type"], inplace=True)
dataset.drop(columns=["cylinder"], inplace=True)
dataset.drop(columns=["transmission_type"], inplace=True)
dataset.drop(columns=["gear_box"], inplace=True)
dataset.drop(columns=["is_front_fog_lights"], inplace=True)
dataset.drop(columns=["is_rear_window_wiper"], inplace=True)
dataset.drop(columns=["is_rear_window_washer"], inplace=True)
dataset.drop(columns=["is_rear_window_defogger"], inplace=True)
dataset.drop(columns=["is_brake_assist"], inplace=True)
dataset.drop(columns=["is_power_door_locks"], inplace=True)
dataset.drop(columns=["is_central_locking"], inplace=True)
dataset.drop(columns=["is_power_steering"], inplace=True)
dataset.drop(columns=["is_driver_seat_height_adjustable"], inplace=True)
dataset.drop(columns=["is_day_night_rear_view_mirror"], inplace=True)
dataset.drop(columns=["is_ecw"], inplace=True)
dataset.drop(columns=["is_speed_alert"], inplace=True)
dataset.drop(columns=["population_density"], inplace=True)
dataset.drop(columns=["make"], inplace=True)
dataset.drop(columns=["turning_radius"], inplace=True)
dataset.drop(columns=["ncap_rating"], inplace=True)
#dataset.drop(columns=["b"], inplace=True)

#dataset.drop(index=dataset.index[:48000], axis=0, inplace=True)

#dataset.dropna(subset=["RainTomorrow"],inplace=True)

#print(list(dataset["Class"].unique()))
#dataset = dataset.drop(dataset[dataset['ReachedOnTime'] == 2].index)

#dataset['ReachedOnTime'] = dataset['ReachedOnTime'].replace(1,'Yes')
#dataset['ReachedOnTime'] = dataset['ReachedOnTime'].replace(0,'No')

#print(list(dataset["Segmentation"].unique()))

dataset.to_csv("datasets/car_insurance1.csv", index=False)