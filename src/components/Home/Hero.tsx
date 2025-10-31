import heroimg from "../../assests/hero-img.jpg";

const bgstyle2 = {
  backgroundImage: `url(${heroimg})`,
  width: "100%",
  backgroundPosition: "center",
  backgroundRepeat: "no-repeat",
  backgroundSize: "cover",

};


const Hero = () => {
  return (
    <>
      <div className="relative z-[-1] py-20  bg-linear-to-r from-amber-200 to-green-500">
        <div className="container py-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 place-items-center gap-4">
            <div className="px-5 order-2 sm:order-1">
              {/*Text-content section*/}
              <div className="text-5xl font-bold text-green-600">
                Eco<span className="text-cyan-950">-Waste AI</span>
              </div>

              <p className="font-bold py-4 text-green-700 text-xl">
                Scan,Upload,Learn - Make Zero Waste
              </p>

              <p className="pb-4 font-semibold text-cyan-950">
                Eco-Waste AI , a website that helps you with identifying the
                different types of waste and with the help of AI will let you
                know if its recyclable or not and how to dispose of it.
              </p>

              <button className="font-bold bg-green-700 text-white rounded py-2 px-5">
                Get Started Now
              </button>
            </div>
            {/*Image section*/}
            <div
              style={bgstyle2}
              className="rounded-3xl overflow-hidden shadow-2xl h-72 sm:h-96 md:h-[350px] order-1 sm:order-2"
            >
              
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Hero


