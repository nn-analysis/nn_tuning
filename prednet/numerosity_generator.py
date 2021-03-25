 
if params.conditionOrder(winPos)==1;
    params.experiment='Dots Area pRF full blanks TR=1.5, nTRs=3';
    params.equalArea = 1;
    dotSizeIn = 3*(7/2)^2*pi;
    dotColors = [0 0 0; 0 0 0];
elseif params.conditionOrder(winPos)==2;
    params.experiment='Dots Size pRF full blanks TR=1.5, nTRs=3';
    params.equalArea = 0;
    dotSize = 7;
    dotColors = [0 0 0; 0 0 0];
elseif params.conditionOrder(winPos)==3;
    params.experiment='Dots Circumference pRF full blanks TR=1.5, nTRs=3';
    params.equalArea = 2;
    dotSizeIn = 19*pi*3;
    dotColors = [0 0 0; 0 0 0];
end

 
if params.equalArea ==1;
    dotSize =(2*(sqrt((dotSizeIn/ndots)/pi)));
    if ndots==2
        recheckDist=5;
    elseif ndots==3
        recheckDist=5;
    elseif ndots==4
        recheckDist=4.8;
    elseif ndots==5
        recheckDist=4.5;
    elseif ndots==6
        recheckDist=4.2;
    elseif ndots==7
        recheckDist=4;
    else
        recheckDist=3;
    end
elseif params.equalArea ==2;
    dotSize = dotSizeIn/ndots/pi;
    if ndots==2
        recheckDist=1.15;
    elseif ndots==3
        recheckDist=1.5;
    elseif ndots==4
        recheckDist=1.9;
    elseif ndots==5
        recheckDist=2.1;
    elseif ndots==6
        recheckDist=2.3;
    elseif ndots==7
        recheckDist=2.5;
    else
        recheckDist=3;
    end

    
elseif params.equalArea ==0;
    if ndots==2
        recheckDist=6;
    elseif ndots==3
        recheckDist=5;
    elseif ndots==4
        recheckDist=4;
    elseif ndots==5
        recheckDist=3.5;
    elseif ndots==6
        recheckDist=3;
    elseif ndots==7
        recheckDist=2.8;
    else
        recheckDist=1.4;
    end
end

 
dotGroup=newDotPattern(ndots,n, dotSize, recheckDist); %n is image size, in pixels. Numbers are set for 73 pixels.

 
Screen('DrawDots',display.windowPtr, double(dotGroup'), double(dotSize), double(dotColors(fixSeqAdder,:)), [display.Rect(1) display.Rect(2)],1);

 

 
function dotGroup=newDotPattern(ndots,n, dotSize, recheckDistance)
if ndots >0;
    recheckCounter=1000;
    while recheckCounter==1000
    for rdots = 1:ndots;
        tempDotPattern = rand(1,2)*n; %Dot position, x,y
        if dotSize>=n
            %GIVE AN ERROR          
        else
            while sqrt((tempDotPattern(1,1)-0.5*n)^2+(tempDotPattern(1,2)-0.5*n)^2)>0.5*n-dotSize/2
                tempDotPattern = rand(1,2)*n;
            end
        end
        A = tempDotPattern(1,1);
        B = tempDotPattern(1,2);

 

 
        if rdots == 1;
            dotGroup = tempDotPattern;
            recheckCounter=1;
        else
            recheck = 1;
            recheckCounter=1;
            while recheck == 1;
                recheck = 0;
                for storedDots = 1:size(dotGroup,1);
                    if recheck == 0;
                        xDist = dotGroup(storedDots,1)-A;
                        yDist = dotGroup(storedDots,2)-B;
                        totalDist = sqrt(xDist^2 + yDist^2);
                        if totalDist < (dotSize * recheckDistance);
                            recheck = 1;
                        end
                    end
                end

 

 
                if recheck == 0;

 
                    dotGroup(rdots,1) = A;
                    dotGroup(rdots,2) = B;
                else
                    tempDotPattern = rand(1,2)*n;

 
                    while sqrt((tempDotPattern(1,1)-0.5*n)^2+(tempDotPattern(1,2)-0.5*n)^2)>0.5*n-dotSize/2
                        tempDotPattern = rand(1,2)*n;
                    end

 
                    A = tempDotPattern(1,1);
                    B = tempDotPattern(1,2);
                    recheckCounter=recheckCounter+1;
                    if recheckCounter==1000
                        dotGroup(rdots,:)=dotGroup(rdots-1,:);
                        %Give an error
                        break;
                    end
                end
            end
        end
    end
    end
else dotGroup = [];
end

 
end
