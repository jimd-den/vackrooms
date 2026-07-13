# hierarchical z-buffer logic prototype
import math

class CoarseZBuffer:
    def __init__(self, width, height, cell_size=16):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = (width + cell_size - 1) // cell_size
        self.rows = (height + cell_size - 1) // cell_size
        self.buffer = [float('inf')] * (self.cols * self.rows)
    
    def test(self, x0, y0, x1, y1, z_near):
        # returns True if node is visible
        c0 = max(0, int(x0) // self.cell_size)
        r0 = max(0, int(y0) // self.cell_size)
        c1 = min(self.cols - 1, int(x1) // self.cell_size)
        r1 = min(self.rows - 1, int(y1) // self.cell_size)
        
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if z_near <= self.buffer[r * self.cols + c]:
                    return True
        return False
        
    def update(self, x0, y0, x1, y1, z_far):
        # Update is tricky. If we just splat a small node, it doesn't cover the whole cell.
        # We can only update the cell's stored depth if the splat COVERs the entire cell?
        # No, a coarse z-buffer stores the FARTHEST depth of any pixel in the cell.
        # Actually, if we track the NEAREST depth written so far per cell...
        pass
