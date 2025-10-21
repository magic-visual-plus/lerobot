# config_tasks.py
TASK_CONFIGS = {
    'libero_goal': {
        'label_names': {
            0: 'background', 1: 'AkitaBlackBowl', 2: 'CreamCheese', 3: 'WineBottle',
            4: 'Plate', 5: 'Woodencabinet', 6: 'FlatStove', 7: 'WineRack',
            8: 'MountedPanda', 9: 'RethinkMount', 10: 'PandaGripper'
        },
        'tasks2labels': {'open_the_middle_drawer_of_the_cabinet': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_bowl_on_the_stove': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_wine_bottle_on_top_of_the_cabinet': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'open_the_top_drawer_and_put_the_bowl_inside': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_bowl_on_top_of_the_cabinet': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'push_the_plate_to_the_front_of_the_stove': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_cream_cheese_in_the_bowl': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'turn_on_the_stove': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_bowl_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_wine_bottle_on_the_rack': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
                        },
        'object_class': {0:'operation_object', 1:'target_object', 2:'gripper', 3:'occupancy'},
        'object2class': {'open_the_middle_drawer_of_the_cabinet': {1:3, 2:3, 3:3, 4:3, 5:0, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'put_the_bowl_on_the_stove': {1:0, 2:3, 3:3, 4:3, 5:3, 6:1, 7:3, 8:3, 9:3, 10:2},
                        'put_the_wine_bottle_on_top_of_the_cabinet': {1:3, 2:3, 3:0, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'open_the_top_drawer_and_put_the_bowl_inside': {1:0, 2:3, 3:3, 4:3, 5:0, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'put_the_bowl_on_top_of_the_cabinet': {1:0, 2:3, 3:3, 4:3, 5:0, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'push_the_plate_to_the_front_of_the_stove': {1:3, 2:3, 3:3, 4:0, 5:3, 6:1, 7:3, 8:3, 9:3, 10:2},
                        'put_the_cream_cheese_in_the_bowl': {1:1, 2:0, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'turn_on_the_stove': {1:3, 2:3, 3:3, 4:3, 5:3, 6:0, 7:3, 8:3, 9:3, 10:2},
                        'put_the_bowl_on_the_plate': {1:0, 2:3, 3:3, 4:1, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'put_the_wine_bottle_on_the_rack': {1:3, 2:3, 3:0, 4:3, 5:3, 6:3, 7:1, 8:3, 9:3, 10:2}
                        }
    },
    'libero_spatial': {
        'label_names': {
            0: 'background', 1: 'AkitaBlackBowl', 2: 'CreamCheese', 3: 'Remekin',
            4: 'Plate', 5: 'Woodencabinet', 6: 'FlatStove', 7: 'WineRack',
            8: 'MountedPanda', 9: 'RethinkMount', 10: 'PandaGripper'
        },
        'tasks2labels': {'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
                        },
        'object_class': {0:'operation_object', 1:'target_object', 2:'gripper', 3:'occupancy'},
        'object2class': {'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate': {1:0, 2:3, 3:3, 4:3, 5:1, 6:3, 7:3, 8:3, 9:3, 10:2}
                        }
    },
    'libero_object': {
        'label_names': {
            0: 'background', 1: 'tomato_sauce', 2: 'basket', 3: 'Remekin',
            4: 'chocolate_pudding', 5: 'orange_juice', 6: 'chocolate_pudding', 7: 'WineRack',
            8: 'MountedPanda', 9: 'RethinkMount', 10: 'PandaGripper'
        },
        'tasks2labels': {'pick_up_the_alphabet_soup_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_cream_cheese_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_salad_dressing_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_bbq_sauce_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_ketchup_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_tomato_sauce_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_butter_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_milk_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_chocolate_pudding_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_orange_juice_and_place_it_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
                        },
        'object_class': {0:'operation_object', 1:'target_object', 2:'gripper', 3:'occupancy'},
        'object2class': {'pick_up_the_alphabet_soup_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_cream_cheese_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_salad_dressing_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_bbq_sauce_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_ketchup_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_tomato_sauce_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_butter_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_milk_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_chocolate_pudding_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
                        'pick_up_the_orange_juice_and_place_it_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2}
                        }
    },
    'libero_10': {
        'label_names': {
            0: 'background', 1: 'tomato_sauce', 2: 'basket', 3: 'Remekin',
            4: 'chocolate_pudding', 5: 'orange_juice', 6: 'chocolate_pudding', 7: 'WineRack',
            8: 'MountedPanda', 9: 'RethinkMount', 10: 'PandaGripper'
        },
        'tasks2labels': {'put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_both_the_cream_cheese_box_and_the_butter_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'turn_on_the_stove_and_put_the_moka_pot_on_it': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_both_moka_pots_on_the_stove': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                        'put_the_yellow_and_white_mug_in_the_microwave_and_close_it': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
                        },
        'object_class': {0:'operation_object', 1:'target_object', 2:'gripper', 3:'occupancy'},
        'object2class': {'put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket': {1:0, 2:3, 3:0, 4:3, 5:3, 6:3, 7:2, 8:1, 9:3, 10:3},
                        'put_both_the_cream_cheese_box_and_the_butter_in_the_basket': {1:3, 2:0, 3:3, 4:3, 5:3, 6:3, 7:0, 8:1, 9:3, 10:3},
                        'turn_on_the_stove_and_put_the_moka_pot_on_it': {1:3, 2:0, 3:0, 4:3, 5:3, 6:2, 7:3, 8:3, 9:3, 10:3},
                        'put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it': {1:0, 2:3, 3:1, 4:3, 5:3, 6:3, 7:2, 8:3, 9:3, 10:3},
                        'put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate': {1:0, 2:3, 3:0, 4:1, 5:3, 6:3, 7:2, 8:3, 9:3, 10:3},
                        'pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy': {1:0, 2:3, 3:1, 4:3, 5:3, 6:2, 7:3, 8:3, 9:3, 10:3},
                        'put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate': {1:0, 2:3, 3:1, 4:3, 5:3, 6:3, 7:2, 8:3, 9:3, 10:3},
                        'put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket': {1:0, 2:0, 3:3, 4:3, 5:1, 6:3, 7:3, 8:2, 9:3, 10:3},
                        'put_both_moka_pots_on_the_stove': {1:0, 2:1, 3:3, 4:3, 5:2, 6:3, 7:3, 8:3, 9:3, 10:3},
                        'put_the_yellow_and_white_mug_in_the_microwave_and_close_it': {1:3, 2:0, 3:1, 4:3, 5:3, 6:2, 7:3, 8:3, 9:3, 10:3}
                        }
    },
    # 'libero_10': {
    #     'label_names': {
    #         0: 'background', 1: 'tomato_sauce', 2: 'basket', 3: 'Remekin',
    #         4: 'chocolate_pudding', 5: 'orange_juice', 6: 'chocolate_pudding', 7: 'WineRack',
    #         8: 'MountedPanda', 9: 'RethinkMount', 10: 'PandaGripper'
    #     },
    #     'tasks2labels': {'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
    #                     'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it': {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}
    #                     },
    #     'object_class': {0:'operation_object', 1:'target_object', 2:'gripper', 3:'occupancy'},
    #     'object2class': {'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2},
    #                     'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it': {1:0, 2:1, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:2}
    #                     }
    # },
}
