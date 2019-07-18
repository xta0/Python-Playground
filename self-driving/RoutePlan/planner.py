
import math

#A* Algorithm
class PathPlanner():
    """Construct a PathPlanner Object"""
    def __init__(self, M, start=None, goal=None):
        """ """
        self.map = M
        self.start= start
        self.goal = goal
        self.closedSet = self.create_closedSet() if goal != None and start != None else None
        self.openSet = self.create_openSet() if goal != None and start != None else None
        self.cameFrom = self.create_cameFrom() if goal != None and start != None else None
        self.gScore = self.create_gScore() if goal != None and start != None else None
        self.fScore = self.create_fScore() if goal != None and start != None else None
        self.path = self.run_search() if self.map and self.start != None and self.goal != None else None
    
    def reconstruct_path(self, current):
        """ Reconstructs path after search """
        total_path = [current]
        while current in self.cameFrom.keys():
            current = self.cameFrom[current]
            total_path.append(current)
        return total_path
    
    def _reset(self):
        """Private method used to reset the closedSet, openSet, cameFrom, gScore, fScore, and path attributes"""
        self.closedSet = None
        self.openSet = None
        self.cameFrom = None
        self.gScore = None
        self.fScore = None
        self.path = self.run_search() if self.map and self.start and self.goal else None

    def run_search(self):
        """ """
        if self.map == None:
            raise(ValueError, "Must create map before running search. Try running PathPlanner.set_map(start_node)")
        if self.goal == None:
            raise(ValueError, "Must create goal node before running search. Try running PathPlanner.set_goal(start_node)")
        if self.start == None:
            raise(ValueError, "Must create start node before running search. Try running PathPlanner.set_start(start_node)")

        self.closedSet = self.closedSet if self.closedSet != None else self.create_closedSet()
        self.openSet = self.openSet if self.openSet != None else  self.create_openSet()
        self.cameFrom = self.cameFrom if self.cameFrom != None else  self.create_cameFrom()
        self.gScore = self.gScore if self.gScore != None else  self.create_gScore()
        self.fScore = self.fScore if self.fScore != None else  self.create_fScore()

        while not self.is_open_empty():
            current = self.get_current_node()

            if current == self.goal:
                self.path = [x for x in reversed(self.reconstruct_path(current))]
                return self.path
            else:
                self.openSet.remove(current)
                self.closedSet.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closedSet:
                    continue    # Ignore the neighbor which is already evaluated.

                if not neighbor in self.openSet:    # Discover a new node
                    self.openSet.add(neighbor)
                
                # The distance from start to a neighbor
                #the "dist_between" function may vary as per the solution requirements.
                if self.get_tentative_gScore(current, neighbor) >= self.get_gScore(neighbor):
                    continue        # This is not a better path.

                # This path is the best until now. Record it!
                self.record_best_path_to(current, neighbor)
        print("No Path Found")
        self.path = None
        return False

    def create_closedSet(self):
        return set()
    
    def create_openSet(self):
        if self.start != None:
            return {self.start}
        return None

    def create_cameFrom(self):
        return {}

    def create_gScore(self):
        gScore = [math.inf] * len(self.map.roads)
        gScore[self.start] = 0.0
        return gScore

    def create_fScore(self):    
        fScore = [math.inf] * len(self.map.roads)
        fScore[self.start] = self.distance(self.start,self.goal)
        return fScore

    def set_map(self, M):
        """Method used to set map attribute """
        self._reset()
        self.start = None
        self.goal = None
        
        self.map = M

    def set_start(self,start):
        self._reset()
        self.start = start

    def set_goal(self, goal):
        self._reset()
        self.goal = goal

    def is_open_empty(self):
        """returns True if the open set is empty. False otherwise. """
        return not len(self.openSet)
    
    def get_current_node(self):
        ans = None
        minv = math.inf
        for node in self.openSet:
            if self.fScore[node] < minv:
                minv = self.fScore[node]
                ans = node
        return ans

    def get_neighbors(self, node):
        return self.map.roads[node]
    
    def get_gScore(self, node):
        return self.gScore[node]
    
    def distance(self,node_1,node_2):
        pt1 = self.map.intersections[node_1]
        pt2 = self.map.intersections[node_2]
        #pt2, pt2 are tupels
        return math.sqrt( (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 )

    def get_tentative_gScore(self, current, neighbor):
        return self.gScore[current] + self.distance(current, neighbor)

    def heuristic_cost_estimate(self, node):
        return self.distance(node,self.goal)
    
    def calculate_fscore(self, node):
        return self.gScore[node] + self.heuristic_cost_estimate(node)

    def record_best_path_to(self, current, neighbor):
        self.cameFrom[neighbor] = current
        self.fScore[current] = self.calculate_fscore(current)
        self.gScore[neighbor] = self.get_tentative_gScore(current,neighbor)
        self.fScore[neighbor] = self.calculate_fscore(neighbor)
        
    