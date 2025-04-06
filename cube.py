class Cube:
    def __init__(self, position, color):
        """
        Initialize a Cube object.

        :param position: A tuple (x, y, z) representing the position of the cube.
        :param color: A string representing the color of the cube.
        """
        self.position = position
        self.color = color

    def __repr__(self):
        return f"Cube(position={self.position}, color='{self.color}')"