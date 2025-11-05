from robosuite.utils.mjcf_utils import new_site
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Libero_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_tabletop_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "light-gray",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "table":
                continue

            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "main_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()
        
        # Example: Custom object positioning
        # You can add manual placement samplers here
        # 
        # from libero.libero.envs.regions.workspace_region_sampler import TableRegionSampler
        # 
        # # Example: Place a specific object at a fixed position
        # if "your_object_name" in self.objects_dict:
        #     custom_sampler = TableRegionSampler(
        #         "your_object_name",
        #         self.objects_dict["your_object_name"],
        #         x_ranges=[[0.1, 0.1]],  # Fixed x position
        #         y_ranges=[[0.2, 0.2]],  # Fixed y position
        #         rotation=(0, 0),  # Fixed rotation
        #         rotation_axis="z",
        #         ensure_object_boundary_in_range=True,
        #         ensure_valid_placement=True,
        #         reference_pos=self.workspace_offset,
        #     )
        #     self.placement_initializer.append_sampler(custom_sampler)

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6586131746834771, 0.0, 1.6103500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )

    def _reset_internal(self):
        """Override to set custom object positions at reset time."""
        super()._reset_internal()
        
        # Example: Set specific object positions after normal reset
        # You can manually set object positions here
        # 
        # # Set position for a specific object
        # if "your_object_name" in self.objects_dict:
        #     obj = self.objects_dict["your_object_name"]
        #     # Set position (x, y, z) and quaternion (w, x, y, z)
        #     custom_pos = [0.1, 0.2, self.table_offset[2] + 0.05]
        #     custom_quat = [1, 0, 0, 0]  # No rotation
        #     
        #     self.sim.data.set_joint_qpos(
        #         obj.joints[-1],
        #         np.concatenate([custom_pos, custom_quat])
        #     )
        
    def set_object_position(self, object_name, position, quaternion=None):
        """Helper method to set object position manually.
        
        Args:
            object_name (str): Name of the object to position
            position (list): [x, y, z] position relative to workspace
            quaternion (list): [w, x, y, z] quaternion rotation (optional)
        """
        if object_name not in self.objects_dict:
            print(f"Warning: Object {object_name} not found in objects_dict")
            return
            
        obj = self.objects_dict[object_name]
        
        # Default quaternion (no rotation)
        if quaternion is None:
            quaternion = [1, 0, 0, 0]
            
        # Adjust position relative to workspace offset
        adjusted_pos = [
            position[0] + self.workspace_offset[0],
            position[1] + self.workspace_offset[1], 
            position[2] + self.workspace_offset[2]
        ]
        
        # Set the joint position
        self.sim.data.set_joint_qpos(
            obj.joints[-1],
            np.concatenate([adjusted_pos, quaternion])
        )
        
    def get_object_position(self, object_name):
        """Get current position of an object.
        
        Args:
            object_name (str): Name of the object
            
        Returns:
            np.array: [x, y, z] position in world coordinates, or None if object not found
        """
        if object_name not in self.obj_body_id:
            print(f"Warning: Object {object_name} not found")
            return None
            
        # Get position from simulation data
        pos = self.sim.data.body_xpos[self.obj_body_id[object_name]]
        return np.array(pos)
    
    def get_object_quaternion(self, object_name):
        """Get current quaternion orientation of an object.
        
        Args:
            object_name (str): Name of the object
            
        Returns:
            np.array: [w, x, y, z] quaternion, or None if object not found
        """
        if object_name not in self.obj_body_id:
            print(f"Warning: Object {object_name} not found")
            return None
            
        # Get quaternion from simulation data
        quat = self.sim.data.body_xquat[self.obj_body_id[object_name]]
        return np.array(quat)
    
    def get_object_pose(self, object_name):
        """Get current position and orientation of an object.
        
        Args:
            object_name (str): Name of the object
            
        Returns:
            dict: {'pos': [x, y, z], 'quat': [w, x, y, z]}, or None if object not found
        """
        if object_name not in self.obj_body_id:
            print(f"Warning: Object {object_name} not found")
            return None
            
        pos = self.sim.data.body_xpos[self.obj_body_id[object_name]]
        quat = self.sim.data.body_xquat[self.obj_body_id[object_name]]
        
        return {
            'pos': np.array(pos),
            'quat': np.array(quat)
        }
    
    def get_all_object_positions(self):
        """Get current positions of all objects in the environment.
        
        Returns:
            dict: Mapping from object_name to position [x, y, z]
        """
        positions = {}
        
        # Get positions for movable objects
        for object_name in self.objects_dict.keys():
            if object_name in self.obj_body_id:
                pos = self.sim.data.body_xpos[self.obj_body_id[object_name]]
                positions[object_name] = np.array(pos)
                
        # Get positions for fixtures
        for fixture_name in self.fixtures_dict.keys():
            if fixture_name in self.obj_body_id:
                pos = self.sim.data.body_xpos[self.obj_body_id[fixture_name]]
                positions[fixture_name] = np.array(pos)
                
        return positions
    
    def get_workspace_offset(self):
        """Get the workspace offset (table center position).
        
        Returns:
            np.array: [x, y, z] workspace offset in world coordinates
        """
        return np.array(self.workspace_offset)
    
    def get_table_info(self):
        """Get complete table information including size and position.
        
        Returns:
            dict: Table information containing offset, size, and other properties
        """
        return {
            'workspace_offset': np.array(self.workspace_offset),
            'table_offset': np.array(self.table_offset),  # Same as workspace_offset
            'table_full_size': np.array(self.table_full_size),
            'table_height': self.workspace_offset[2],
            'table_surface_z': self.workspace_offset[2] + self.table_full_size[2]/2,
            'workspace_name': self.workspace_name,
            'z_offset': self.z_offset
        }
    
    def get_world_position_from_relative(self, relative_position):
        """Convert relative position (relative to table center) to world coordinates.
        
        Args:
            relative_position (list or np.array): [x, y, z] position relative to table center
            
        Returns:
            np.array: [x, y, z] position in world coordinates
        """
        relative_pos = np.array(relative_position)
        workspace_offset = np.array(self.workspace_offset)
        return relative_pos + workspace_offset
    
    def get_relative_position_from_world(self, world_position):
        """Convert world coordinates to relative position (relative to table center).
        
        Args:
            world_position (list or np.array): [x, y, z] position in world coordinates
            
        Returns:
            np.array: [x, y, z] position relative to table center
        """
        world_pos = np.array(world_position)
        workspace_offset = np.array(self.workspace_offset)
        return world_pos - workspace_offset
