from helper.maths import inside, segment_intersect
from shapes.rigid import Rigid
from glm import vec2


def sutherland_hodgmen(subject: Rigid, clip: Rigid) -> list[vec2]:
    output_list = subject.vertices
    
    for clip_start, clip_end in clip.edges:
        input_list = output_list[:]
        output_list = []
        
        if len(input_list) == 0:
            break
        
        s = input_list[-1]
        for e in input_list:
            if inside(e, clip_start, clip_end):
                if not inside(s, clip_start, clip_end):
                    # s is outside, e is inside so add intersection
                    i = segment_intersect(s, e, clip_start, clip_end)
                    if i is not None:
                        output_list.append(i)
                output_list.append(e)
            elif inside(s, clip_start, clip_end):
                # s is inside, e is outside so add intersection
                i = segment_intersect(s, e, clip_start, clip_end)
                if i is not None:
                    output_list.append(i)
            s = e
            
    return output_list