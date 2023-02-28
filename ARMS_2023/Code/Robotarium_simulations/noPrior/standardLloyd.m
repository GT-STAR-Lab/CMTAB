% Coverage control using Lloyd's Algorithm.


%standardlloyd(8, [2 2 3 1], [0.15 0.1 0.45 0.2]);
%% Experiment Constants
function [myx, species] = standardLloyd(numRobots, robot_per_species, sensing_radii, visualise)
%Run the simulation for a specific number of iterations
    %visualise = false;
    iterations = 2000;
    %% Set up the Robotarium object
    %robot_per_species = [2 2 3 1];
    %sensing_radii = [0.15 0.1 0.45 0.2];
    N = numRobots;
    species = whichspecies(robot_per_species, N);

    x_init = generate_initial_conditions(N,'Width',1.1,'Height',1.1,'Spacing', 0.17);
    %Spacing sould be 0.35 for running in robotarium?
    x_init = x_init - [min(x_init(1,:)) - (-1.6 + 0.2);min(x_init(2,:)) - (-1 + 0.2);0];
    r = Robotarium('NumberOfRobots', N, 'ShowFigure', visualise ,'InitialConditions',x_init);

    %Initialize velocity vector
    dxi = zeros(2, N);
    
    %Boundary
    crs = [r.boundaries(1), r.boundaries(3);
           r.boundaries(1), r.boundaries(4);
           r.boundaries(2), r.boundaries(4);
           r.boundaries(2), r.boundaries(3)];
    
    %% 
    
    % crs = [r.boundaries(1), r.boundaries(3);
    %        r.boundaries(1), r.boundaries(4);
    %        1/3*r.boundaries(1), r.boundaries(4);
    %        1/3*r.boundaries(1), 0;
    %        1/3*r.boundaries(2), 0;
    %        1/3*r.boundaries(2), r.boundaries(4);
    %        r.boundaries(2), r.boundaries(4);
    %        r.boundaries(2), r.boundaries(3)];
    %% Grab tools we need to convert from single-integrator to unicycle dynamics
    
    % Single-integrator -> unicycle dynamics mapping
    [~, uni_to_si_states] = create_si_to_uni_mapping();
    si_to_uni_dyn = create_si_to_uni_dynamics();
    % Single-integrator barrier certificates
    uni_barrier_cert_boundary = create_uni_barrier_certificate_with_boundary();
    % Single-integrator position controller
    motion_controller = create_si_position_controller('XVelocityGain', 0.8, 'YVelocityGain', 0.8, 'VelocityMagnitudeLimit', 0.1);
    x = r.get_poses();
    verCellHandle = zeros(N,1);
    %% Plotting Setup
    if (visualise == true)
        marker_size = determine_marker_size(r, 0.08);
  
        %cellColors = cool(N);
        for i = 1:N % color according to robot
            %verCellHandle(i)  = patch(x(1,i),x(2,i),cellColors(i,:),'FaceAlpha', 0.3); % use color i  -- no robot assigned yet
            verCellHandle(i)  = patch(x(1,i),x(2,i),'white','FaceAlpha', 0.3, 'EdgeColor', 'white'); % use color i  -- no robot assigned yet
            hold on
        end
        pathHandle = zeros(N,1);      
        for i = 1:N % color according to
            %pathHandle(i)  = plot(x(1,i),x(2,i),'-.','color',cellColors(i,:)*.9, 'LineWidth',4);
            pathHandle(i)  = plot(x(1,i),x(2,i),'-.','color','white', 'LineWidth',4);
        end
        %centroidHandle = plot(x(1,:),x(2,:),'+','MarkerSize',marker_size, 'LineWidth',2, 'Color', 'k');
        centroidHandle = plot(x(1,:),x(2,:),'+','MarkerSize',marker_size, 'LineWidth',2, 'Color', 'w');
        
        for i = 1:N % color according to
            xD = [get(pathHandle(i),'XData'),x(1,i)];
            yD = [get(pathHandle(i),'YData'),x(2,i)];
            set(pathHandle(i),'XData',xD,'YData',yD);%plot path position
        end
    end

    r.step();
    %% Main Loop
    
    for t = 1:iterations
        
        % Retrieve the most recent poses from the Robotarium.  The time delay is
        % approximately 0.033 seconds
        x = r.get_poses();
    
        % Convert to SI states
        xi = uni_to_si_states(x);
        
        %% Algorithm
        [Px, Py] = lloydsAlgorithm(x(1,:)',x(2,:)', crs, verCellHandle, visualise);
        dxi = motion_controller(x(1:2, :), [Px';Py']);
             
        %% Avoid actuator errors
        
        % To avoid errors, we need to threshold dxi
        norms = arrayfun(@(x) norm(dxi(:, x)), 1:N);
        threshold = 3/4*r.max_linear_velocity;
        to_thresh = norms > threshold;
        dxi(:, to_thresh) = threshold*dxi(:, to_thresh)./norms(to_thresh);
        
        %% Use barrier certificate and convert to unicycle dynamics
        dxu = si_to_uni_dyn(dxi, x);
        dxu = uni_barrier_cert_boundary(dxu, x);
        
        %% Send velocities to agents
        
        %Set velocities
        r.set_velocities(1:N, dxu);
        if (visualise == true)
            %% Update Plot Handles
            for i = 1:N % color according to
               xD = [get(pathHandle(i),'XData'),x(1,i)];
               yD = [get(pathHandle(i),'YData'),x(2,i)];
               set(pathHandle(i),'XData',xD,'YData',yD);%plot path position
               
            end
        
        set(centroidHandle,'XData',Px,'YData',Py);%plot centroid position
        end

        %Iterate experiment
        r.step();
    end

    if (visualise == true)

        for i = 1:N
            xD = [get(pathHandle(i),'XData'),x(1,i)];
            yD = [get(pathHandle(i),'YData'),x(2,i)];
            rad = cell2mat(sensing_radii(species(1, i)));
            %hold on
            th = 0:pi/50:2*pi;
            xunit = rad * cos(th) + xD(1,iterations);
            yunit = rad * sin(th) + yD(1,iterations);
            %plot(xunit, yunit);
            fill(xunit, yunit, 'b', 'FaceAlpha',0.03);
        end
    

    pause(20)
    end

    % We can call this function to debug our experiment!  Fix all the errors
    % before submitting to maximize the chance that your experiment runs
    % successfully.
    r.debug();
    
    %% Helper Functions


    function [xspecies] = whichspecies(robot_per_species, N)
        rng('shuffle');
        p = randperm(N,N);
        xspecies = zeros(1,N);
        i = 1;
        for j = 1:4
            for num = 1:cell2mat(robot_per_species(j))
                xspecies(p(i)) = j;
                i = i+1;
            end
        end
    
    end
    
    function [ poses ] = generate_initial_conditions(N, varargin)
% GENERATE_INITIAL_CONDITIONS generate random poses in a circle
% The default parameter values are correctly sized for the Robotarium's
% physical testbed.
%
%   GENERATE_INITIAL_CONDITIONS(5) generates 3 x 5 matrix of
%   random poses
%
%   GENERATE_INITIAL_CONDITIONS(5, 'Spacing', 0.2, 'Width',
%   3.2, 'Height', 2) generates 3 x 5 matrix of random poses with
%   spacing 0.2 m in a rectangle of 3.2 m width and 2 m height.
%
%   Example:
%      poses = generate_initial_conditions(5);
%   
%   Notes:
%       N should be a positive integer.
    
    poses = zeros(3, N);
    
    parser = inputParser;
    parser.addParameter('Spacing', 0.3);
    parser.addParameter('Width', 3.0);
    parser.addParameter('Height', 1.8);
    parse(parser, varargin{:});
    
    spacing = parser.Results.Spacing;
    width = parser.Results.Width;
    height = parser.Results.Height;

    numX = floor(width / spacing);
    numY = floor(height / spacing);
    values = randperm(numX * numY, N);

    for i = 1:N
       [x, y] = ind2sub([numX numY], values(i));
       x = x*spacing - (width/2); 
       y = y*spacing - (height/2);
       poses(1:2, i) = [x ; y];
    end
    
    poses(3, :) = (rand(1, N)*2*pi - pi);
end


    % Marker Size Helper Function to scale size with figure window
    % Input: robotarium instance, desired size of the marker in meters
    function marker_size = determine_marker_size(robotarium_instance, marker_size_meters)
    
        % Get the size of the robotarium figure window in pixels
        curunits = get(robotarium_instance.figure_handle, 'Units');
        set(robotarium_instance.figure_handle, 'Units', 'Points');
        cursize = get(robotarium_instance.figure_handle, 'Position');
        set(robotarium_instance.figure_handle, 'Units', curunits);
        
        % Determine the ratio of the robot size to the x-axis (the axis are
        % normalized so you could do this with y and figure height as well).
        marker_ratio = (marker_size_meters)/(robotarium_instance.boundaries(2) -...
            robotarium_instance.boundaries(1));
        
        % Determine the marker size in points so it fits the window. cursize(3) is
        % the width of the figure window in pixels. (the axis are
        % normalized so you could do this with y and figure height as well).
        marker_size = cursize(3) * marker_ratio;
        
   end
    
    
    %% Lloyds algorithm put together by Aaron Becker
    
    function [Px, Py] = lloydsAlgorithm(Px,Py, crs, verCellHandle, visualise)
        % LLOYDSALGORITHM runs Lloyd's algorithm on the particles at xy positions 
        % (Px,Py) within the boundary polygon crs for numIterations iterations
        % showPlot = true will display the results graphically.  
        % 
        % Lloyd's algorithm starts with an initial distribution of samples or
        % points and consists of repeatedly executing one relaxation step:
        %   1.  The Voronoi diagram of all the points is computed.
        %   2.  Each cell of the Voronoi diagram is integrated and the centroid is computed.
        %   3.  Each point is then moved to the centroid of its Voronoi cell.
        %
        % Inspired by http://www.mathworks.com/matlabcentral/fileexchange/34428-voronoilimit
        % Requires the Polybool function of the mapping toolbox to run.
        %
        % Run with no input to see example.  To initialize a square with 50 robots 
        % in left middle, run:
        %lloydsAlgorithm(0.01*rand(50,1),zeros(50,1)+1/2, [0,0;0,1;1,1;1,0], 200, true)
        %
        % Made by: Aaron Becker, atbecker@uh.edu
        format compact
    
        % initialize random generator in repeatable fashion
        sd = 20;
        rng(sd)
    
    %         crs = [ 0, 0;    
    %             0, yrange;
    %             1/3*xrange, yrange;  % a world with a narrow passage
    %             1/3*xrange, 1/4*yrange;
    %             2/3*xrange, 1/4*yrange;
    %             2/3*xrange, yrange;
    %             xrange, yrange;
    %             xrange, 0];
    
        xrange = max(crs(:,1)) - min(crs(:,1));
        yrange = max(crs(:,2)) - min(crs(:,2));
    
        % Apply LLYOD's Algorithm
        [v,c]=VoronoiBounded(Px,Py,crs);
    
        for i = 1:numel(c) %calculate the centroid of each cell
            [cx,cy] = PolyCentroid(v(c{i},1),v(c{i},2));
            cx = min(max(crs(:,1)),max(min(crs(:,1)), cx));
            cy = min(max(crs(:,2)),max(min(crs(:,2)), cy));
            if ~isnan(cx) && inpolygon(cx,cy,crs(:,1),crs(:,2))
                Px(i) = cx;  %don't update if goal is outside the polygon
                Py(i) = cy;
            end
        end
        if (visualise == true)
            for i = 1:numel(c) % update Voronoi cells
                set(verCellHandle(i), 'XData',v(c{i},1),'YData',v(c{i},2));
            end
        end
    end
        
    function [Cx,Cy] = PolyCentroid(X,Y)
        % POLYCENTROID returns the coordinates for the centroid of polygon with vertices X,Y
        % The centroid of a non-self-intersecting closed polygon defined by n vertices (x0,y0), (x1,y1), ..., (xn?1,yn?1) is the point (Cx, Cy), where
        % In these formulas, the vertices are assumed to be numbered in order of their occurrence along the polygon's perimeter, and the vertex ( xn, yn ) is assumed to be the same as ( x0, y0 ). Note that if the points are numbered in clockwise order the area A, computed as above, will have a negative sign; but the centroid coordinates will be correct even in this case.http://en.wikipedia.org/wiki/Centroid
        % A = polyarea(X,Y)
    
        Xa = [X(2:end);X(1)];
        Ya = [Y(2:end);Y(1)];
    
        A = 1/2*sum(X.*Ya-Xa.*Y); %signed area of the polygon
    
        Cx = (1/(6*A)*sum((X + Xa).*(X.*Ya-Xa.*Y)));
        Cy = (1/(6*A)*sum((Y + Ya).*(X.*Ya-Xa.*Y)));
    end
    
    function [V,C]=VoronoiBounded(x,y, crs)
        % VORONOIBOUNDED computes the Voronoi cells about the points (x,y) inside
        % the bounding box (a polygon) crs.  If crs is not supplied, an
        % axis-aligned box containing (x,y) is used.
    
        bnd=[min(x) max(x) min(y) max(y)]; %data bounds
        if nargin < 3
            crs=double([bnd(1) bnd(4);bnd(2) bnd(4);bnd(2) bnd(3);bnd(1) bnd(3);bnd(1) bnd(4)]);
        end
    
        rgx = max(crs(:,1))-min(crs(:,1));
        rgy = max(crs(:,2))-min(crs(:,2));
        rg = max(rgx,rgy);
        midx = (max(crs(:,1))+min(crs(:,1)))/2;
        midy = (max(crs(:,2))+min(crs(:,2)))/2;
    
        % add 4 additional edges
        xA = [x; midx + [0;0;-5*rg;+5*rg]];
        yA = [y; midy + [-5*rg;+5*rg;0;0]];
    
        [vi,ci]=voronoin([xA,yA]);
    
        % remove the last 4 cells
        C = ci(1:end-4);
        V = vi;
        % use Polybool to crop the cells
        %Polybool for restriction of polygons to domain.
    
        for ij=1:length(C)
                % thanks to http://www.mathworks.com/matlabcentral/fileexchange/34428-voronoilimit
                % first convert the contour coordinate to clockwise order:
                [X2, Y2] = poly2cw_custom(V(C{ij},1),V(C{ij},2));
                tempA = polyshape(crs(:,1), crs(:,2),'Simplify',false);
                tempB = polyshape(X2, Y2,'Simplify',false);
                tempC = intersect(tempA,tempB);
                [xb, yb] = boundary(tempC);
                %[xb, yb] = polybool('intersection',crs(:,1),crs(:,2),X2,Y2);
                ix=nan(1,length(xb));
                for il=1:length(xb)
                    if any(V(:,1)==xb(il)) && any(V(:,2)==yb(il))
                        ix1=find(V(:,1)==xb(il));
                        ix2=find(V(:,2)==yb(il));
                        for ib=1:length(ix1)
                            if any(ix1(ib)==ix2)
                                ix(il)=ix1(ib);
                            end
                        end
                        if isnan(ix(il))==1
                            lv=length(V);
                            V(lv+1,1)=xb(il);
                            V(lv+1,2)=yb(il);
                            ix(il)=lv+1;
                        end
                    else
                        lv=length(V);
                        V(lv+1,1)=xb(il);
                        V(lv+1,2)=yb(il);
                        ix(il)=lv+1;
                    end
                end
                C{ij}=ix;
    
        end
    end
    
    function [ordered_x, ordered_y] = poly2cw_custom(x,y)
        cx = mean(x);
        cy = mean(y);
        a = atan2(y-cy, x -cx);
        
        [~, order] = sort(a);
        ordered_x = x(order);
        ordered_y = y(order);
    end

myx = [r.get_poses(); species];
end
