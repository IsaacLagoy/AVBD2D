def find_max_bounds(arr, idcs, thresh) -> None:
    # constants
    l = len(arr)
    
    # loop checks
    is_in = arr[0] >= thresh
    
    # final checks
    end_found = False
    begin_found = False
    
    for i in range(1, l):
        c = arr[i]

        if c >= thresh:
            
            # we are entering the new threshold
            if not is_in:
                idcs[0] = i - 1
                is_in = True
                begin_found = True
                
            continue
        
        # below threshold
        # check to see if we are leaving
        if is_in:
            is_in = False
            idcs[1] = i - 1
            end_found = True
            
    # edge case - end is at 0 [----++++]
    if not end_found:
        idcs[1] = l - 1
        
    # edge case - beginning is at 0 [++++----]
    # element before begin is -1 
    if not begin_found:
        idcs[0] = l - 1
        

a = [ 0.20815219,  0.2944445,  -0.20815219, -0.2944445]
t_a = -1.0965769290924072

b = [-0.31775182, -0.75031066, -1.0965769,  -0.66401815]
t_b =  0.2944445013999939 



        
buffer = [-1, -1]
vals = [1, -1, -1, 1]
thresh = 0

# find_max_bounds(vals, buffer, thresh)

# print(buffer)

find_max_bounds(a, buffer, t_a)

print(buffer)

find_max_bounds(b, buffer, t_b)

print(buffer)