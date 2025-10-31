
import image2 from "../../assests/image2.jpg";

const bgstyle2 = {
  backgroundImage: `url(${image2})`,
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",
};
const Part1 = () => {
  return (
    <>
      <div className="relative z-[-1] py-20  bg-linear-to-r from-amber-200 to-green-500">
        <div className="container py-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 place-items-center">
            <div className="px-5 order-2 sm:order-1 ">
              <div className="text-5xl font-bold text-green-600 ">
                Effortless <span className="text-cyan-950">Waste Identification</span> 
              </div>
              <p className="font-semibold text-cyan-950 py-4">
                Our smart system simplifies waste sorting and disposal, making
                it easy for everyone to contribute to a greener planet.
                Understand how every step works to ensure effective waste
                management.
              </p>

              <button className= "font-bold bg-green-700 text-white rounded py-2 px-5">Get Started Now</button>
            </div>
            <div style={bgstyle2} className="rounded-3xl overflow-hidden shadow-2xl h-72 sm:96 md:h-[350px] order-1 sm:order-2 ">
              
            </div>

          </div>
        </div>
      </div>
    </>
  );
};

export default Part1;